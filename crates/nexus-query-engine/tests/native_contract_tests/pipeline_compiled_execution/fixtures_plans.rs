use super::*;
use crate::projection_outputs;

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
    );
    let aggregate = builder.add_aggregate(
        join,
        AggregateSpec::new(
            vec![Expr::column("left_id")],
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("amount"), "amount_sum")],
        ),
        Arc::clone(&aggregate_schema),
    );
    let sort = builder.add_sort(
        aggregate,
        SortSpec::new(vec![SortExpr::new(Expr::column("amount_sum"), true, false)]),
        Arc::clone(&aggregate_schema),
    );
    let projection = builder.add_projection(
        sort,
        projection_outputs(vec![Expr::column("left_id"), Expr::column("amount_sum")], aggregate_schema),
    );
    builder.finish_with_host_sink(projection)
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
    );
    let aggregate = builder.add_aggregate(
        join,
        AggregateSpec::new(
            vec![Expr::column("k")],
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("amount"), "amount_sum")],
        ),
        Arc::clone(&aggregate_schema),
    );
    let sort = builder.add_sort(
        aggregate,
        SortSpec::new(vec![SortExpr::new(Expr::column("amount_sum"), true, false)]),
        aggregate_schema,
    );
    builder.finish_with_host_sink(sort)
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
    let window_schema = Arc::new(Schema::new(vec![
        Field::new("item_sk", DataType::Int32, false),
        Field::new("class_id", DataType::Int32, false),
        Field::new("category_id", DataType::Int32, false),
        Field::new("amount_sum", DataType::Int64, true),
        Field::new("class_amount_sum", DataType::Int64, true),
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
    );
    let pre_aggregate = builder.add_aggregate(
        first_join,
        AggregateSpec::new(
            vec![Expr::column("item_sk")],
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("amount"), "pre_amount_sum")],
        ),
        Arc::clone(&pre_aggregate_schema),
    );
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
    );
    let aggregate = builder.add_aggregate(
        second_join,
        AggregateSpec::new(
            vec![Expr::column("item_sk"), Expr::column("class_id"), Expr::column("category_id")],
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("pre_amount_sum"), "amount_sum")],
        ),
        Arc::clone(&aggregate_schema),
    );
    let window_input_sort = builder.add_sort(
        aggregate,
        SortSpec::new(vec![SortExpr::new(Expr::column("class_id"), false, false)]),
        Arc::clone(&aggregate_schema),
    );
    let window = builder.add_window(
        window_input_sort,
        WindowSpec::new(
            vec![Expr::column("class_id")],
            Vec::new(),
            vec![WindowFunction::aggregate(AggregateFunction::Sum, Expr::column("amount_sum"), "class_amount_sum")],
        ),
        Arc::clone(&window_schema),
    );
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
    );
    let final_sort = builder.add_sort(
        projection,
        SortSpec::new(vec![
            SortExpr::new(Expr::column("category_id"), false, true),
            SortExpr::new(Expr::column("class_id"), false, true),
            SortExpr::new(Expr::column("item_sk"), false, true),
            SortExpr::new(Expr::column("class_amount_sum"), false, true),
        ]),
        projection_schema,
    );
    builder.finish_with_host_sink(final_sort)
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
    );
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
    );
    let aggregate = builder.add_aggregate(
        second_join,
        AggregateSpec::new(
            vec![Expr::column("k")],
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("amount"), "amount_sum")],
        ),
        Arc::clone(&aggregate_schema),
    );
    let sort = builder.add_sort(
        aggregate,
        SortSpec::new(vec![SortExpr::new(Expr::column("amount_sum"), true, false)]),
        aggregate_schema,
    );
    builder.finish_with_host_sink(sort)
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
    );
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
    );
    let aggregate = builder.add_aggregate(
        second_join,
        AggregateSpec::new(
            vec![Expr::column("k")],
            vec![AggregateExpr::new(AggregateFunction::StddevSamp, Expr::column("amount"), "amount_stddev")],
        ),
        Arc::clone(&aggregate_schema),
    );
    let sort = builder.add_sort(
        aggregate,
        SortSpec::new(vec![SortExpr::new(Expr::column("amount_stddev"), true, false)]),
        aggregate_schema,
    );
    builder.finish_with_host_sink(sort)
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
    );
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
    );
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
    );
    let aggregate = builder.add_aggregate(
        second_join,
        AggregateSpec::new(
            vec![Expr::column("k")],
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("amount"), "amount_sum")],
        ),
        Arc::clone(&aggregate_schema),
    );
    let sort = builder.add_sort(
        aggregate,
        SortSpec::new(vec![SortExpr::new(Expr::column("amount_sum"), true, false)]),
        aggregate_schema,
    );
    builder.finish_with_host_sink(sort)
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
    );
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
    );
    let grouped = builder.add_aggregate(
        grouped_join,
        AggregateSpec::new(
            vec![Expr::column("k")],
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("amount"), "amount_sum")],
        ),
        Arc::clone(&grouped_schema),
    );
    let grouped_projection = builder.add_projection(
        grouped,
        projection_outputs(vec![Expr::column("k"), Expr::column("amount_sum")], grouped_schema),
    );
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
    );
    let scalar = builder.add_aggregate(
        scalar_join,
        AggregateSpec::new(
            Vec::new(),
            vec![AggregateExpr::new(AggregateFunction::Avg, Expr::column("amount"), "amount_avg")],
        ),
        Arc::clone(&scalar_schema),
    );
    let scalar_projection =
        builder.add_projection(scalar, projection_outputs(vec![Expr::column("amount_avg")], scalar_schema));
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
    );
    let sort = builder.add_sort(
        root_join,
        SortSpec::new(vec![SortExpr::new(Expr::column("amount_sum"), true, false)]),
        root_join_schema,
    );
    builder.finish_with_host_sink(sort)
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
    );
    let aggregate = builder.add_aggregate(
        join,
        AggregateSpec::new(
            vec![Expr::column("k")],
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("amount"), "amount_sum")],
        ),
        Arc::clone(&aggregate_schema),
    );
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
    );
    let sort = builder.add_sort(
        self_join,
        SortSpec::new(vec![SortExpr::new(Expr::column("left_amount_sum"), true, false)]),
        self_join_schema,
    );
    builder.finish_with_host_sink(sort)
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

pub(super) fn large_full_scan_rollup_rank_topk_plan() -> QueryPlan {
    rollup_rank_topk_plan_with_fact_file(
        LocalParquetFile::new("/tmp/compiled-large-rollup-rank-topk-fact.parquet").with_file_size_bytes(2_u64 << 30),
        true,
    )
}

pub(super) fn large_full_scan_rollup_rank_native_fragment_plan() -> QueryPlan {
    rollup_rank_topk_plan_with_fact_file(
        LocalParquetFile::new("/tmp/compiled-large-rollup-rank-native-fragment-fact.parquet")
            .with_file_size_bytes(2_u64 << 30),
        false,
    )
}

pub(super) fn selected_large_rollup_rank_topk_plan() -> QueryPlan {
    rollup_rank_topk_plan_with_fact_file(
        LocalParquetFile::new("/tmp/compiled-selected-large-rollup-rank-topk-fact.parquet")
            .with_file_size_bytes(4_u64 << 30)
            .with_row_groups(vec![0, 1])
            .with_projected_row_group_bytes(
                vec![768 << 20, 768 << 20, 768 << 20, 768 << 20],
                vec![768 << 20, 768 << 20, 768 << 20, 768 << 20],
            )
            .with_row_group_rows(vec![10_000_000; 4]),
        true,
    )
}

pub(super) fn selected_q86_class_rollup_rank_topk_plan() -> QueryPlan {
    let row_group_bytes = vec![2 << 20; 448];
    rollup_rank_topk_plan_with_fact_file(
        LocalParquetFile::new("/tmp/compiled-selected-q86-class-rollup-rank-topk-fact.parquet")
            .with_file_size_bytes(2_u64 << 30)
            .with_row_groups((0..448).collect())
            .with_projected_row_group_bytes(row_group_bytes.clone(), row_group_bytes)
            .with_row_group_rows(vec![100_000; 448]),
        true,
    )
}

pub(super) fn selected_q70_class_left_semi_rollup_rank_topk_plan() -> QueryPlan {
    let row_group_bytes = vec![2 << 20; 448];
    rollup_rank_topk_plan_with_fact_file_and_join_type(
        LocalParquetFile::new("/tmp/compiled-selected-q70-class-rollup-rank-topk-fact.parquet")
            .with_file_size_bytes(2_u64 << 30)
            .with_row_groups((0..448).collect())
            .with_projected_row_group_bytes(row_group_bytes.clone(), row_group_bytes)
            .with_row_group_rows(vec![100_000; 448]),
        true,
        JoinType::LeftSemi,
    )
}

pub(super) fn selected_q67_class_rollup_rank_topk_projection_chain_plan() -> QueryPlan {
    let row_group_bytes = vec![2 << 20; 448];
    rollup_rank_topk_plan_with_projection_chain(
        LocalParquetFile::new("/tmp/compiled-selected-q67-class-rollup-rank-topk-fact.parquet")
            .with_file_size_bytes(2_u64 << 30)
            .with_row_groups((0..448).collect())
            .with_projected_row_group_bytes(row_group_bytes.clone(), row_group_bytes)
            .with_row_group_rows(vec![100_000; 448]),
    )
}

pub(super) fn selected_wide_q67_class_rollup_rank_topk_plan() -> QueryPlan {
    let row_group_bytes = vec![2 << 20; 448];
    wide_rollup_rank_topk_plan_with_fact_file(
        LocalParquetFile::new("/tmp/compiled-selected-wide-q67-class-rollup-rank-topk-fact.parquet")
            .with_file_size_bytes(2_u64 << 30)
            .with_row_groups((0..448).collect())
            .with_projected_row_group_bytes(row_group_bytes.clone(), row_group_bytes)
            .with_row_group_rows(vec![100_000; 448]),
        false,
        false,
    )
}

pub(super) fn selected_too_wide_rollup_rank_topk_plan() -> QueryPlan {
    let row_group_bytes = vec![2 << 20; 448];
    wide_rollup_rank_topk_plan_with_fact_file(
        LocalParquetFile::new("/tmp/compiled-selected-too-wide-rollup-rank-topk-fact.parquet")
            .with_file_size_bytes(2_u64 << 30)
            .with_row_groups((0..448).collect())
            .with_projected_row_group_bytes(row_group_bytes.clone(), row_group_bytes)
            .with_row_group_rows(vec![100_000; 448]),
        true,
        false,
    )
}

pub(super) fn selected_wide_non_rollup_grouping_sets_rank_topk_plan() -> QueryPlan {
    let row_group_bytes = vec![2 << 20; 448];
    wide_rollup_rank_topk_plan_with_fact_file(
        LocalParquetFile::new("/tmp/compiled-selected-wide-non-rollup-rank-topk-fact.parquet")
            .with_file_size_bytes(2_u64 << 30)
            .with_row_groups((0..448).collect())
            .with_projected_row_group_bytes(row_group_bytes.clone(), row_group_bytes)
            .with_row_group_rows(vec![100_000; 448]),
        false,
        true,
    )
}

pub(super) fn selected_small_rollup_rank_topk_plan() -> QueryPlan {
    rollup_rank_topk_plan_with_fact_file(
        LocalParquetFile::new("/tmp/compiled-selected-small-rollup-rank-topk-fact.parquet")
            .with_file_size_bytes(4_u64 << 30)
            .with_row_groups(vec![0])
            .with_projected_row_group_bytes(
                vec![128 << 20, 2 << 30, 2 << 30, 2 << 30],
                vec![128 << 20, 2 << 30, 2 << 30, 2 << 30],
            )
            .with_row_group_rows(vec![1_000_000; 4]),
        true,
    )
}

pub(super) fn statistical_join_final_full_sort_plan() -> QueryPlan {
    let fact_schema = Arc::new(Schema::new(vec![
        Field::new("date_sk", DataType::Int32, false),
        Field::new("item_sk", DataType::Int32, false),
        Field::new("warehouse_sk", DataType::Int32, false),
        Field::new("qty", DataType::Float64, true),
    ]));
    let item_schema = Arc::new(Schema::new(vec![Field::new("item_sk", DataType::Int32, false)]));
    let warehouse_schema = Arc::new(Schema::new(vec![
        Field::new("warehouse_sk", DataType::Int32, false),
        Field::new("warehouse_name", DataType::Utf8, true),
    ]));
    let date_schema = Arc::new(Schema::new(vec![
        Field::new("date_sk", DataType::Int32, false),
        Field::new("month", DataType::Int32, false),
    ]));
    let joined_schema = Arc::new(Schema::new(vec![
        Field::new("item_sk", DataType::Int32, false),
        Field::new("warehouse_sk", DataType::Int32, false),
        Field::new("warehouse_name", DataType::Utf8, true),
        Field::new("month", DataType::Int32, false),
        Field::new("qty", DataType::Float64, true),
    ]));
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("warehouse_name", DataType::Utf8, true),
        Field::new("warehouse_sk", DataType::Int32, false),
        Field::new("item_sk", DataType::Int32, false),
        Field::new("month", DataType::Int32, false),
        Field::new("stdev", DataType::Float64, true),
        Field::new("mean", DataType::Float64, true),
    ]));
    let final_join_schema = Arc::new(Schema::new(vec![
        Field::new("wsk1", DataType::Int32, false),
        Field::new("isk1", DataType::Int32, false),
        Field::new("month1", DataType::Int32, false),
        Field::new("mean1", DataType::Float64, true),
        Field::new("cov1", DataType::Float64, true),
        Field::new("warehouse_sk", DataType::Int32, false),
        Field::new("item_sk", DataType::Int32, false),
        Field::new("month", DataType::Int32, false),
        Field::new("mean", DataType::Float64, true),
        Field::new("cov", DataType::Float64, true),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let first_branch = statistical_inventory_branch(
        &mut builder,
        "/tmp/compiled-q39-class-inventory-m01.parquet",
        "/tmp/compiled-q39-class-item.parquet",
        "/tmp/compiled-q39-class-warehouse.parquet",
        "/tmp/compiled-q39-class-date-m01.parquet",
        Arc::clone(&fact_schema),
        Arc::clone(&item_schema),
        Arc::clone(&warehouse_schema),
        Arc::clone(&date_schema),
        Arc::clone(&joined_schema),
        Arc::clone(&aggregate_schema),
    );
    let second_branch = statistical_inventory_branch(
        &mut builder,
        "/tmp/compiled-q39-class-inventory-m02.parquet",
        "/tmp/compiled-q39-class-item.parquet",
        "/tmp/compiled-q39-class-warehouse.parquet",
        "/tmp/compiled-q39-class-date-m02.parquet",
        fact_schema,
        item_schema,
        warehouse_schema,
        date_schema,
        joined_schema,
        aggregate_schema,
    );
    let joined = builder.add_join(
        first_branch,
        second_branch,
        JoinSpec::new(
            JoinType::Inner,
            vec![
                JoinKey::new(
                    ColumnRef::input(0, "warehouse_sk", "warehouse_sk"),
                    ColumnRef::input(1, "warehouse_sk", "warehouse_sk"),
                ),
                JoinKey::new(ColumnRef::input(0, "item_sk", "item_sk"), ColumnRef::input(1, "item_sk", "item_sk")),
            ],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "warehouse_sk", "warehouse_sk"), "wsk1"),
            JoinOutputColumn::left(ColumnRef::input(0, "item_sk", "item_sk"), "isk1"),
            JoinOutputColumn::left(ColumnRef::input(0, "month", "month"), "month1"),
            JoinOutputColumn::left(ColumnRef::input(0, "mean", "mean"), "mean1"),
            JoinOutputColumn::left(ColumnRef::input(0, "cov", "cov"), "cov1"),
            JoinOutputColumn::right(ColumnRef::input(1, "warehouse_sk", "warehouse_sk"), "warehouse_sk"),
            JoinOutputColumn::right(ColumnRef::input(1, "item_sk", "item_sk"), "item_sk"),
            JoinOutputColumn::right(ColumnRef::input(1, "month", "month"), "month"),
            JoinOutputColumn::right(ColumnRef::input(1, "mean", "mean"), "mean"),
            JoinOutputColumn::right(ColumnRef::input(1, "cov", "cov"), "cov"),
        ])
        .with_build_side(JoinSide::Right),
        Arc::clone(&final_join_schema),
    );
    let sort = builder.add_sort(
        joined,
        SortSpec::new(vec![
            SortExpr::new(Expr::column("wsk1"), false, true),
            SortExpr::new(Expr::column("isk1"), false, true),
            SortExpr::new(Expr::column("mean1"), false, true),
            SortExpr::new(Expr::column("cov1"), false, true),
            SortExpr::new(Expr::column("mean"), false, true),
            SortExpr::new(Expr::column("cov"), false, true),
        ]),
        final_join_schema,
    );
    builder.finish_with_host_sink(sort)
}

fn rollup_rank_topk_plan_with_fact_file(fact_file: LocalParquetFile, include_final_topk: bool) -> QueryPlan {
    rollup_rank_topk_plan_with_fact_file_and_join_type(fact_file, include_final_topk, JoinType::Inner)
}

#[allow(clippy::too_many_arguments)]
fn statistical_inventory_branch(
    builder: &mut QueryPlanBuilder,
    inventory_path: &str,
    item_path: &str,
    warehouse_path: &str,
    date_path: &str,
    fact_schema: Arc<Schema>,
    item_schema: Arc<Schema>,
    warehouse_schema: Arc<Schema>,
    date_schema: Arc<Schema>,
    joined_schema: Arc<Schema>,
    aggregate_schema: Arc<Schema>,
) -> NodeId {
    let fact = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        inventory_path,
        fact_schema,
        vec!["date_sk".into(), "item_sk".into(), "warehouse_sk".into(), "qty".into()],
    )));
    let item = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        item_path,
        item_schema,
        vec!["item_sk".into()],
    )));
    let warehouse = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        warehouse_path,
        warehouse_schema,
        vec!["warehouse_sk".into(), "warehouse_name".into()],
    )));
    let date = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        date_path,
        date_schema,
        vec!["date_sk".into(), "month".into()],
    )));

    let item_join = builder.add_join(
        fact,
        item,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "item_sk", "item_sk"), ColumnRef::input(1, "item_sk", "item_sk"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "date_sk", "date_sk"), "date_sk"),
            JoinOutputColumn::left(ColumnRef::input(0, "warehouse_sk", "warehouse_sk"), "warehouse_sk"),
            JoinOutputColumn::left(ColumnRef::input(0, "item_sk", "item_sk"), "item_sk"),
            JoinOutputColumn::left(ColumnRef::input(0, "qty", "qty"), "qty"),
        ])
        .with_build_side(JoinSide::Right),
        Arc::new(Schema::new(vec![
            Field::new("date_sk", DataType::Int32, false),
            Field::new("warehouse_sk", DataType::Int32, false),
            Field::new("item_sk", DataType::Int32, false),
            Field::new("qty", DataType::Float64, true),
        ])),
    );
    let warehouse_join = builder.add_join(
        item_join,
        warehouse,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "warehouse_sk", "warehouse_sk"),
                ColumnRef::input(1, "warehouse_sk", "warehouse_sk"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "date_sk", "date_sk"), "date_sk"),
            JoinOutputColumn::left(ColumnRef::input(0, "item_sk", "item_sk"), "item_sk"),
            JoinOutputColumn::left(ColumnRef::input(0, "warehouse_sk", "warehouse_sk"), "warehouse_sk"),
            JoinOutputColumn::right(ColumnRef::input(1, "warehouse_name", "warehouse_name"), "warehouse_name"),
            JoinOutputColumn::left(ColumnRef::input(0, "qty", "qty"), "qty"),
        ])
        .with_build_side(JoinSide::Right),
        Arc::new(Schema::new(vec![
            Field::new("date_sk", DataType::Int32, false),
            Field::new("item_sk", DataType::Int32, false),
            Field::new("warehouse_sk", DataType::Int32, false),
            Field::new("warehouse_name", DataType::Utf8, true),
            Field::new("qty", DataType::Float64, true),
        ])),
    );
    let date_join = builder.add_join(
        warehouse_join,
        date,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "date_sk", "date_sk"), ColumnRef::input(1, "date_sk", "date_sk"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "item_sk", "item_sk"), "item_sk"),
            JoinOutputColumn::left(ColumnRef::input(0, "warehouse_sk", "warehouse_sk"), "warehouse_sk"),
            JoinOutputColumn::left(ColumnRef::input(0, "warehouse_name", "warehouse_name"), "warehouse_name"),
            JoinOutputColumn::right(ColumnRef::input(1, "month", "month"), "month"),
            JoinOutputColumn::left(ColumnRef::input(0, "qty", "qty"), "qty"),
        ])
        .with_build_side(JoinSide::Right),
        Arc::clone(&joined_schema),
    );
    let aggregate = builder.add_aggregate(
        date_join,
        AggregateSpec::new(
            vec![
                Expr::column("warehouse_name"),
                Expr::column("warehouse_sk"),
                Expr::column("item_sk"),
                Expr::column("month"),
            ],
            vec![
                AggregateExpr::new(AggregateFunction::StddevSamp, Expr::column("qty"), "stdev"),
                AggregateExpr::new(AggregateFunction::Avg, Expr::column("qty"), "mean"),
            ],
        ),
        aggregate_schema,
    );
    let filter = builder.add_filter(
        aggregate,
        Expr::binary(
            Expr::binary(Expr::column("stdev"), BinaryOp::Divide, Expr::column("mean")),
            BinaryOp::Gt,
            Expr::literal(LiteralValue::Float64(1.0)),
        ),
        Arc::new(Schema::new(vec![
            Field::new("warehouse_name", DataType::Utf8, true),
            Field::new("warehouse_sk", DataType::Int32, false),
            Field::new("item_sk", DataType::Int32, false),
            Field::new("month", DataType::Int32, false),
            Field::new("stdev", DataType::Float64, true),
            Field::new("mean", DataType::Float64, true),
        ])),
    );
    builder.add_projection(
        filter,
        vec![
            ProjectionOutput::new(Expr::column("warehouse_sk"), Field::new("warehouse_sk", DataType::Int32, false)),
            ProjectionOutput::new(Expr::column("item_sk"), Field::new("item_sk", DataType::Int32, false)),
            ProjectionOutput::new(Expr::column("month"), Field::new("month", DataType::Int32, false)),
            ProjectionOutput::new(Expr::column("mean"), Field::new("mean", DataType::Float64, true)),
            ProjectionOutput::new(
                Expr::binary(Expr::column("stdev"), BinaryOp::Divide, Expr::column("mean")).alias("cov"),
                Field::new("cov", DataType::Float64, true),
            ),
        ],
    )
}

fn wide_rollup_rank_topk_plan_with_fact_file(
    fact_file: LocalParquetFile,
    include_extra_rollup_key: bool,
    use_non_rollup_grouping_masks: bool,
) -> QueryPlan {
    let mut fact_fields = vec![
        Field::new("id", DataType::Int32, false),
        Field::new("d_year", DataType::Int32, false),
        Field::new("d_qoy", DataType::Int32, false),
        Field::new("d_moy", DataType::Int32, false),
        Field::new("s_store_id", DataType::Int32, false),
        Field::new("amount", DataType::Int64, false),
    ];
    if include_extra_rollup_key {
        fact_fields.insert(5, Field::new("channel_id", DataType::Int32, false));
    }
    let mut fact_projection =
        vec!["id".into(), "d_year".into(), "d_qoy".into(), "d_moy".into(), "s_store_id".into(), "amount".into()];
    if include_extra_rollup_key {
        fact_projection.insert(5, "channel_id".into());
    }
    let fact_schema = Arc::new(Schema::new(fact_fields));
    let dim_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("category", DataType::Utf8, true),
        Field::new("class", DataType::Utf8, true),
        Field::new("brand", DataType::Utf8, true),
        Field::new("product", DataType::Utf8, true),
    ]));

    let mut output_fields = vec![
        Field::new("category", DataType::Utf8, true),
        Field::new("class", DataType::Utf8, true),
        Field::new("brand", DataType::Utf8, true),
        Field::new("product", DataType::Utf8, true),
        Field::new("d_year", DataType::Int32, false),
        Field::new("d_qoy", DataType::Int32, false),
        Field::new("d_moy", DataType::Int32, false),
        Field::new("s_store_id", DataType::Int32, false),
    ];
    if include_extra_rollup_key {
        output_fields.push(Field::new("channel_id", DataType::Int32, false));
    }
    output_fields.push(Field::new("amount", DataType::Int64, false));
    let join_schema = Arc::new(Schema::new(output_fields.clone()));

    let mut aggregate_fields =
        output_fields.iter().filter(|field| field.name() != "amount").cloned().collect::<Vec<_>>();
    aggregate_fields
        .extend([Field::new("__grouping_id", DataType::UInt32, false), Field::new("sumsales", DataType::Int64, true)]);
    let aggregate_schema = Arc::new(Schema::new(aggregate_fields.clone()));
    let mut ranked_fields = aggregate_fields;
    ranked_fields.push(Field::new("rk", DataType::UInt64, false));
    let ranked_schema = Arc::new(Schema::new(ranked_fields));

    let mut builder = QueryPlanBuilder::new();
    let fact = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![fact_file],
        fact_schema,
        fact_projection,
    )));
    let dim = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/compiled-wide-rollup-rank-topk-dim.parquet",
        dim_schema,
        vec!["id".into(), "category".into(), "class".into(), "brand".into(), "product".into()],
    )));
    let mut join_outputs = vec![
        JoinOutputColumn::right(ColumnRef::input(1, "category", "category"), "category"),
        JoinOutputColumn::right(ColumnRef::input(1, "class", "class"), "class"),
        JoinOutputColumn::right(ColumnRef::input(1, "brand", "brand"), "brand"),
        JoinOutputColumn::right(ColumnRef::input(1, "product", "product"), "product"),
        JoinOutputColumn::left(ColumnRef::input(0, "d_year", "d_year"), "d_year"),
        JoinOutputColumn::left(ColumnRef::input(0, "d_qoy", "d_qoy"), "d_qoy"),
        JoinOutputColumn::left(ColumnRef::input(0, "d_moy", "d_moy"), "d_moy"),
        JoinOutputColumn::left(ColumnRef::input(0, "s_store_id", "s_store_id"), "s_store_id"),
    ];
    if include_extra_rollup_key {
        join_outputs.push(JoinOutputColumn::left(ColumnRef::input(0, "channel_id", "channel_id"), "channel_id"));
    }
    join_outputs.push(JoinOutputColumn::left(ColumnRef::input(0, "amount", "amount"), "amount"));
    let join = builder.add_join(
        fact,
        dim,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "id", "id"), ColumnRef::input(1, "id", "id"))],
        )
        .with_output_projection(join_outputs)
        .with_build_side(JoinSide::Right),
        Arc::clone(&join_schema),
    );
    let mut group_exprs = vec![
        Expr::column("category"),
        Expr::column("class"),
        Expr::column("brand"),
        Expr::column("product"),
        Expr::column("d_year"),
        Expr::column("d_qoy"),
        Expr::column("d_moy"),
        Expr::column("s_store_id"),
    ];
    if include_extra_rollup_key {
        group_exprs.push(Expr::column("channel_id"));
    }
    let grouping_sets = if use_non_rollup_grouping_masks {
        let mut sets = vec![vec![false; group_exprs.len()]];
        for idx in 0..group_exprs.len() {
            let mut mask = vec![false; group_exprs.len()];
            mask[idx] = true;
            sets.push(mask);
        }
        sets
    } else {
        (0..=group_exprs.len())
            .map(|rolled_count| {
                (0..group_exprs.len()).map(|idx| idx >= group_exprs.len() - rolled_count).collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    };
    let aggregate = builder.add_aggregate(
        join,
        AggregateSpec::new(
            group_exprs,
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("amount"), "sumsales")],
        )
        .with_grouping_sets(GroupingSetsSpec::new(grouping_sets, "__grouping_id")),
        Arc::clone(&aggregate_schema),
    );
    let window_input_sort = builder.add_sort(
        aggregate,
        SortSpec::new(vec![SortExpr::new(Expr::column("sumsales"), true, false)]),
        Arc::clone(&aggregate_schema),
    );
    let window = builder.add_window(
        window_input_sort,
        WindowSpec::new(
            vec![Expr::column("category")],
            vec![SortExpr::new(Expr::column("sumsales"), true, false)],
            vec![WindowFunction::rank("rk")],
        ),
        Arc::clone(&ranked_schema),
    );
    let filter = builder.add_filter(
        window,
        Expr::binary(Expr::column("rk"), BinaryOp::LtEq, Expr::literal(LiteralValue::UInt64(100))),
        Arc::clone(&ranked_schema),
    );
    let final_sort = builder.add_sort(
        filter,
        SortSpec::new(vec![
            SortExpr::new(Expr::column("category"), false, true),
            SortExpr::new(Expr::column("class"), false, true),
            SortExpr::new(Expr::column("brand"), false, true),
            SortExpr::new(Expr::column("product"), false, true),
            SortExpr::new(Expr::column("d_year"), false, true),
            SortExpr::new(Expr::column("d_qoy"), false, true),
            SortExpr::new(Expr::column("d_moy"), false, true),
            SortExpr::new(Expr::column("s_store_id"), false, true),
            SortExpr::new(Expr::column("sumsales"), true, false),
        ]),
        Arc::clone(&ranked_schema),
    );
    let limit = builder.add_limit(final_sort, LimitSpec::fetch(100), ranked_schema);
    builder.finish_with_host_sink(limit)
}

/// q67-class shape: aggregate over a projection chain that computes the
/// measure (`amount * quantity`, then COALESCE), with group keys on both sides
/// of the join. Mirrors TPC-DS q67's
/// `Aggregate(rollup) <- Projection(coalesce) <- Projection(arith) <- Join`
/// shape that the projection-chain peeling in `aggregate_join_pushdown` must
/// handle.
fn rollup_rank_topk_plan_with_projection_chain(fact_file: LocalParquetFile) -> QueryPlan {
    let fact_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("left_group", DataType::Utf8, true),
        Field::new("quantity", DataType::Int64, false),
        Field::new("amount", DataType::Int64, false),
    ]));
    let dim_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("right_group", DataType::Utf8, true),
    ]));
    let join_schema = Arc::new(Schema::new(vec![
        Field::new("left_group", DataType::Utf8, true),
        Field::new("quantity", DataType::Int64, false),
        Field::new("amount", DataType::Int64, false),
        Field::new("right_group", DataType::Utf8, true),
    ]));
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("left_group", DataType::Utf8, true),
        Field::new("right_group", DataType::Utf8, true),
        Field::new("__grouping_id", DataType::UInt32, false),
        Field::new("sumsales", DataType::Int64, true),
    ]));
    let ranked_schema = Arc::new(Schema::new(vec![
        Field::new("left_group", DataType::Utf8, true),
        Field::new("right_group", DataType::Utf8, true),
        Field::new("__grouping_id", DataType::UInt32, false),
        Field::new("sumsales", DataType::Int64, true),
        Field::new("rk", DataType::UInt64, false),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let fact = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![fact_file],
        fact_schema,
        vec!["id".into(), "left_group".into(), "quantity".into(), "amount".into()],
    )));
    let dim = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/compiled-q67-class-rollup-rank-topk-dim.parquet",
        dim_schema,
        vec!["id".into(), "right_group".into()],
    )));
    let join = builder.add_join(
        fact,
        dim,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "id", "id"), ColumnRef::input(1, "id", "id"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "left_group", "left_group"), "left_group"),
            JoinOutputColumn::left(ColumnRef::input(0, "quantity", "quantity"), "quantity"),
            JoinOutputColumn::left(ColumnRef::input(0, "amount", "amount"), "amount"),
            JoinOutputColumn::right(ColumnRef::input(1, "right_group", "right_group"), "right_group"),
        ])
        .with_build_side(JoinSide::Right),
        Arc::clone(&join_schema),
    );
    let common_projection = builder.add_projection(
        join,
        vec![
            ProjectionOutput::new(Expr::column("left_group"), Field::new("left_group", DataType::Utf8, true)),
            ProjectionOutput::new(Expr::column("right_group"), Field::new("right_group", DataType::Utf8, true)),
            ProjectionOutput::new(
                Expr::binary(Expr::column("amount"), BinaryOp::Multiply, Expr::column("quantity"))
                    .alias("__common_expr_1"),
                Field::new("__common_expr_1", DataType::Int64, true),
            ),
        ],
    );
    let coalesce_projection = builder.add_projection(
        common_projection,
        vec![
            ProjectionOutput::new(Expr::column("left_group"), Field::new("left_group", DataType::Utf8, true)),
            ProjectionOutput::new(Expr::column("right_group"), Field::new("right_group", DataType::Utf8, true)),
            ProjectionOutput::new(
                Expr::column("__common_expr_1"),
                Field::new("__common_expr_1", DataType::Int64, true),
            ),
            ProjectionOutput::new(
                Expr::Case {
                    branches: vec![CaseWhen::new(
                        Expr::IsNotNull(Box::new(Expr::column("__common_expr_1"))),
                        Expr::column("__common_expr_1"),
                    )],
                    else_expr: Box::new(Expr::literal(LiteralValue::Int64(0))),
                }
                .alias("__cudf_aggregate_arg_0"),
                Field::new("__cudf_aggregate_arg_0", DataType::Int64, false),
            ),
        ],
    );
    let aggregate = builder.add_aggregate(
        coalesce_projection,
        AggregateSpec::new(
            vec![Expr::column("left_group"), Expr::column("right_group")],
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("__cudf_aggregate_arg_0"), "sumsales")],
        )
        .with_grouping_sets(GroupingSetsSpec::new(
            vec![vec![false, false], vec![false, true], vec![true, true]],
            "__grouping_id",
        )),
        Arc::clone(&aggregate_schema),
    );
    let window_input_sort = builder.add_sort(
        aggregate,
        SortSpec::new(vec![SortExpr::new(Expr::column("sumsales"), true, false)]),
        Arc::clone(&aggregate_schema),
    );
    let window = builder.add_window(
        window_input_sort,
        WindowSpec::new(
            vec![Expr::column("left_group")],
            vec![SortExpr::new(Expr::column("sumsales"), true, false)],
            vec![WindowFunction::rank("rk")],
        ),
        Arc::clone(&ranked_schema),
    );
    let filter = builder.add_filter(
        window,
        Expr::binary(Expr::column("rk"), BinaryOp::LtEq, Expr::literal(LiteralValue::UInt64(100))),
        Arc::clone(&ranked_schema),
    );
    let final_sort = builder.add_sort(
        filter,
        SortSpec::new(vec![
            SortExpr::new(Expr::column("left_group"), false, true),
            SortExpr::new(Expr::column("right_group"), false, true),
            SortExpr::new(Expr::column("sumsales"), true, false),
        ]),
        Arc::clone(&ranked_schema),
    );
    let limit = builder.add_limit(final_sort, LimitSpec::fetch(100), ranked_schema);
    builder.finish_with_host_sink(limit)
}

fn rollup_rank_topk_plan_with_fact_file_and_join_type(
    fact_file: LocalParquetFile,
    include_final_topk: bool,
    join_type: JoinType,
) -> QueryPlan {
    let fact_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("category", DataType::Utf8, true),
        Field::new("class", DataType::Utf8, true),
        Field::new("brand", DataType::Utf8, true),
        Field::new("amount", DataType::Int64, false),
    ]));
    let dim_schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
    let join_schema = Arc::new(Schema::new(vec![
        Field::new("category", DataType::Utf8, true),
        Field::new("class", DataType::Utf8, true),
        Field::new("brand", DataType::Utf8, true),
        Field::new("amount", DataType::Int64, false),
    ]));
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("category", DataType::Utf8, true),
        Field::new("class", DataType::Utf8, true),
        Field::new("brand", DataType::Utf8, true),
        Field::new("__grouping_id", DataType::UInt32, false),
        Field::new("sumsales", DataType::Int64, true),
    ]));
    let ranked_schema = Arc::new(Schema::new(vec![
        Field::new("category", DataType::Utf8, true),
        Field::new("class", DataType::Utf8, true),
        Field::new("brand", DataType::Utf8, true),
        Field::new("__grouping_id", DataType::UInt32, false),
        Field::new("sumsales", DataType::Int64, true),
        Field::new("rk", DataType::UInt64, false),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let fact = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![fact_file],
        fact_schema,
        vec!["id".into(), "category".into(), "class".into(), "brand".into(), "amount".into()],
    )));
    let dim = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/compiled-large-rollup-rank-topk-dim.parquet",
        dim_schema,
        vec!["id".into()],
    )));
    let join = builder.add_join(
        fact,
        dim,
        JoinSpec::new(join_type, vec![JoinKey::new(ColumnRef::input(0, "id", "id"), ColumnRef::input(1, "id", "id"))])
            .with_output_projection(vec![
                JoinOutputColumn::left(ColumnRef::input(0, "category", "category"), "category"),
                JoinOutputColumn::left(ColumnRef::input(0, "class", "class"), "class"),
                JoinOutputColumn::left(ColumnRef::input(0, "brand", "brand"), "brand"),
                JoinOutputColumn::left(ColumnRef::input(0, "amount", "amount"), "amount"),
            ])
            .with_build_side(JoinSide::Right),
        Arc::clone(&join_schema),
    );
    let aggregate = builder.add_aggregate(
        join,
        AggregateSpec::new(
            vec![Expr::column("category"), Expr::column("class"), Expr::column("brand")],
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("amount"), "sumsales")],
        )
        .with_grouping_sets(GroupingSetsSpec::new(
            vec![vec![false, false, false], vec![false, false, true], vec![false, true, true], vec![true, true, true]],
            "__grouping_id",
        )),
        Arc::clone(&aggregate_schema),
    );
    let window_input_sort = builder.add_sort(
        aggregate,
        SortSpec::new(vec![SortExpr::new(Expr::column("sumsales"), true, false)]),
        Arc::clone(&aggregate_schema),
    );
    let window = builder.add_window(
        window_input_sort,
        WindowSpec::new(
            vec![Expr::column("category")],
            vec![SortExpr::new(Expr::column("sumsales"), true, false)],
            vec![WindowFunction::rank("rk")],
        ),
        Arc::clone(&ranked_schema),
    );
    let filter = builder.add_filter(
        window,
        Expr::binary(Expr::column("rk"), BinaryOp::LtEq, Expr::literal(LiteralValue::UInt64(100))),
        Arc::clone(&ranked_schema),
    );
    if !include_final_topk {
        return builder.finish_with_host_sink(filter);
    }
    let final_sort = builder.add_sort(
        filter,
        SortSpec::new(vec![
            SortExpr::new(Expr::column("category"), false, true),
            SortExpr::new(Expr::column("class"), false, true),
            SortExpr::new(Expr::column("brand"), false, true),
            SortExpr::new(Expr::column("sumsales"), true, false),
        ]),
        Arc::clone(&ranked_schema),
    );
    let limit = builder.add_limit(final_sort, LimitSpec::fetch(100), ranked_schema);
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

pub(super) fn projection_root_global_limit_join_aggregate_plan() -> QueryPlan {
    projection_root_global_limit_join_aggregate_plan_with_variant(ProjectionGlobalLimitPlanVariant::Standard)
}

pub(super) fn limit_root_projected_global_hash_join_aggregate_with_sibling_plan() -> QueryPlan {
    let fact_schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("d", DataType::Int32, false),
        Field::new("c", DataType::Int32, false),
        Field::new("v", DataType::Int32, true),
    ]));
    let date_schema = Arc::new(Schema::new(vec![Field::new("d", DataType::Int32, false)]));
    let category_schema = Arc::new(Schema::new(vec![Field::new("c", DataType::Int32, false)]));
    let first_join_schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("c", DataType::Int32, false),
        Field::new("v", DataType::Int32, true),
    ]));
    let base_join_schema =
        Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false), Field::new("v", DataType::Int32, true)]));
    let sibling_left_schema =
        Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false), Field::new("s", DataType::Int32, false)]));
    let sibling_right_schema = Arc::new(Schema::new(vec![Field::new("s", DataType::Int32, false)]));
    let sibling_join_schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false)]));
    let sibling_aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("sibling_count", DataType::Int64, false),
    ]));
    let sibling_projection_schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false)]));
    let root_join_schema =
        Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false), Field::new("v", DataType::Int32, true)]));
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("metric", DataType::Int64, false),
    ]));
    let projection_schema = Arc::new(Schema::new(vec![
        Field::new("metric", DataType::Int64, false),
        Field::new("k", DataType::Int32, false),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let fact = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/projection-global-limit-sibling-fact.parquet",
        fact_schema,
        vec!["k".into(), "d".into(), "c".into(), "v".into()],
    )));
    let date = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/projection-global-limit-sibling-date.parquet",
        date_schema,
        vec!["d".into()],
    )));
    let first_join = builder.add_join(
        fact,
        date,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "d", "d"), ColumnRef::input(1, "d", "d"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "k", "k"), "k"),
            JoinOutputColumn::left(ColumnRef::input(0, "c", "c"), "c"),
            JoinOutputColumn::left(ColumnRef::input(0, "v", "v"), "v"),
        ])
        .with_build_side(JoinSide::Right),
        Arc::clone(&first_join_schema),
    );
    let category = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/projection-global-limit-sibling-category.parquet",
        category_schema,
        vec!["c".into()],
    )));
    let base_join = builder.add_join(
        first_join,
        category,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "c", "c"), ColumnRef::input(1, "c", "c"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "k", "k"), "k"),
            JoinOutputColumn::left(ColumnRef::input(0, "v", "v"), "v"),
        ])
        .with_build_side(JoinSide::Right),
        Arc::clone(&base_join_schema),
    );

    let sibling_left = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/projection-global-limit-sibling-left.parquet",
        sibling_left_schema,
        vec!["k".into(), "s".into()],
    )));
    let sibling_right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/projection-global-limit-sibling-right.parquet",
        sibling_right_schema,
        vec!["s".into()],
    )));
    let sibling_join = builder.add_join(
        sibling_left,
        sibling_right,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "s", "s"), ColumnRef::input(1, "s", "s"))],
        )
        .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "k", "k"), "k")])
        .with_build_side(JoinSide::Right),
        Arc::clone(&sibling_join_schema),
    );
    let sibling_aggregate = builder.add_aggregate(
        sibling_join,
        AggregateSpec::new(
            vec![Expr::column("k")],
            vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("k"), "sibling_count")],
        ),
        Arc::clone(&sibling_aggregate_schema),
    );
    let sibling_projection = builder.add_projection(
        sibling_aggregate,
        projection_outputs(vec![Expr::column("k")], Arc::clone(&sibling_projection_schema)),
    );

    let root_join = builder.add_join(
        base_join,
        sibling_projection,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "k", "k"), ColumnRef::input(1, "k", "k"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "k", "k"), "k"),
            JoinOutputColumn::left(ColumnRef::input(0, "v", "v"), "v"),
        ])
        .with_build_side(JoinSide::Right),
        Arc::clone(&root_join_schema),
    );
    let aggregate = builder.add_aggregate(
        root_join,
        AggregateSpec::new(
            vec![Expr::column("k")],
            vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("v"), "metric")],
        ),
        Arc::clone(&aggregate_schema),
    );
    let projection = builder.add_projection(
        aggregate,
        projection_outputs(vec![Expr::column("metric"), Expr::column("k")], Arc::clone(&projection_schema)),
    );
    let limit = builder.add_limit(projection, LimitSpec::fetch(10), projection_schema);
    builder.finish_with_host_sink(limit)
}

#[allow(dead_code)]
pub(super) fn projection_root_scalar_fanout_plan() -> QueryPlan {
    projection_root_scalar_fanout_plan_with_variant(ProjectionScalarSetopPlanVariant::ScalarFanout)
}

#[allow(dead_code)]
pub(super) fn projection_root_scalar_fanout_one_row_driver_plan() -> QueryPlan {
    projection_root_scalar_fanout_plan_with_variant(ProjectionScalarSetopPlanVariant::OneRowDriver)
}

#[allow(dead_code)]
pub(super) fn projection_root_scalar_fanout_scan_limit_driver_plan() -> QueryPlan {
    projection_root_scalar_fanout_plan_with_variant(ProjectionScalarSetopPlanVariant::ScanLimitDriver)
}

#[allow(dead_code)]
pub(super) fn projection_root_scalar_fanout_max_rows_driver_plan() -> QueryPlan {
    projection_root_scalar_fanout_plan_with_variant(ProjectionScalarSetopPlanVariant::MaxRowsDriver)
}

#[allow(dead_code)]
pub(super) fn projection_root_scalar_fanout_limit_driver_plan() -> QueryPlan {
    projection_root_scalar_fanout_plan_with_variant(ProjectionScalarSetopPlanVariant::LimitDriver)
}

#[allow(dead_code)]
pub(super) fn projection_root_scalar_fanout_unsupported_join_plan() -> QueryPlan {
    projection_root_scalar_fanout_plan_with_variant(ProjectionScalarSetopPlanVariant::UnsupportedJoin)
}

#[allow(dead_code)]
pub(super) fn projection_root_repeated_scalar_fanout_plan() -> QueryPlan {
    projection_root_repeated_scalar_fanout_plan_with_variant(ProjectionScalarSetopPlanVariant::RepeatedScalarFanout)
}

#[allow(dead_code)]
pub(super) fn projection_root_repeated_scalar_fanout_unbounded_input_plan() -> QueryPlan {
    projection_root_repeated_scalar_fanout_plan_with_variant(
        ProjectionScalarSetopPlanVariant::RepeatedScalarFanoutUnboundedInput,
    )
}

#[allow(dead_code)]
pub(super) fn projection_root_setop_dedup_scalar_plan() -> QueryPlan {
    projection_root_setop_dedup_scalar_plan_with_variant(ProjectionScalarSetopPlanVariant::SetopDedup)
}

#[allow(dead_code)]
pub(super) fn projection_root_setop_dedup_input_plan() -> QueryPlan {
    projection_root_setop_dedup_scalar_plan_with_variant(ProjectionScalarSetopPlanVariant::SetopDedupInput)
}

#[allow(dead_code)]
pub(super) fn projection_root_setop_dedup_scalar_non_dedup_left_anti_input_plan() -> QueryPlan {
    projection_root_setop_dedup_scalar_plan_with_variant(ProjectionScalarSetopPlanVariant::NonDedupLeftAntiInput)
}

#[allow(dead_code)]
pub(super) fn projection_root_setop_distinct_aggregate_plan() -> QueryPlan {
    projection_root_setop_dedup_scalar_plan_with_variant(ProjectionScalarSetopPlanVariant::DistinctAggregate)
}

#[allow(dead_code)]
pub(super) fn projection_root_keyless_grouped_aggregate_input_plan() -> QueryPlan {
    let fact_schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("amount", DataType::Int64, true),
    ]));
    let grouped_schema =
        Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false), Field::new("cnt", DataType::Int64, false)]));
    let avg_schema = Arc::new(Schema::new(vec![Field::new("avg_amount", DataType::Float64, true)]));
    let join_schema = Arc::new(Schema::new(vec![
        Field::new("cnt", DataType::Int64, false),
        Field::new("avg_amount", DataType::Float64, true),
    ]));
    let output_schema = Arc::new(Schema::new(vec![Field::new("bucket", DataType::Float64, true)]));

    let mut builder = QueryPlanBuilder::new();
    let fact = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/projection-keyless-grouped-input.parquet",
        fact_schema,
        vec!["k".into(), "amount".into()],
    )));
    let grouped = builder.add_aggregate(
        fact,
        AggregateSpec::new(
            vec![Expr::column("k")],
            vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("amount"), "cnt")],
        ),
        Arc::clone(&grouped_schema),
    );
    let avg = builder.add_aggregate(
        fact,
        AggregateSpec::new(
            Vec::new(),
            vec![AggregateExpr::new(AggregateFunction::Avg, Expr::column("amount"), "avg_amount")],
        ),
        Arc::clone(&avg_schema),
    );
    let cross = builder.add_join(
        grouped,
        avg,
        JoinSpec::new(JoinType::Cross, Vec::new())
            .with_output_projection(vec![
                JoinOutputColumn::left(ColumnRef::input(0, "cnt", "cnt"), "cnt"),
                JoinOutputColumn::right(ColumnRef::input(1, "avg_amount", "avg_amount"), "avg_amount"),
            ])
            .with_build_side(JoinSide::Right),
        Arc::clone(&join_schema),
    );
    let projection = builder.add_projection(
        cross,
        projection_outputs(
            vec![Expr::Case {
                branches: vec![CaseWhen::new(
                    Expr::binary(Expr::column("cnt"), BinaryOp::Gt, Expr::literal(LiteralValue::Int64(0))),
                    Expr::column("avg_amount"),
                )],
                else_expr: Box::new(Expr::literal(LiteralValue::Float64(0.0))),
            }],
            output_schema,
        ),
    );
    builder.finish_with_host_sink(projection)
}

#[allow(dead_code)]
pub(super) fn projection_root_scalar_join_aggregate_input_plan() -> QueryPlan {
    let fact_schema = Arc::new(Schema::new(vec![
        Field::new("fact_k", DataType::Int32, false),
        Field::new("dim_a", DataType::Int32, false),
        Field::new("dim_b", DataType::Int32, false),
    ]));
    let dim_schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false)]));
    let joined_schema = Arc::new(Schema::new(vec![Field::new("fact_k", DataType::Int32, false)]));
    let presence_schema = Arc::new(Schema::new(vec![Field::new("__row_present", DataType::Int32, false)]));
    let count_schema = Arc::new(Schema::new(vec![Field::new("cnt", DataType::Int64, false)]));

    let mut builder = QueryPlanBuilder::new();
    let fact = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/projection-scalar-join-aggregate-input-fact.parquet",
        fact_schema,
        vec!["fact_k".into(), "dim_a".into(), "dim_b".into()],
    )));
    let dim_a = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/projection-scalar-join-aggregate-input-dim-a.parquet",
        Arc::clone(&dim_schema),
        vec!["k".into()],
    )));
    let dim_b = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/projection-scalar-join-aggregate-input-dim-b.parquet",
        Arc::clone(&dim_schema),
        vec!["k".into()],
    )));
    let joined_a = builder.add_join(
        fact,
        dim_a,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "dim_a", "dim_a"), ColumnRef::input(1, "k", "k"))],
        )
        .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "fact_k", "fact_k"), "fact_k")])
        .with_build_side(JoinSide::Right),
        Arc::clone(&joined_schema),
    );
    let joined_b = builder.add_join(
        joined_a,
        dim_b,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "fact_k", "fact_k"), ColumnRef::input(1, "k", "k"))],
        )
        .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "fact_k", "fact_k"), "fact_k")])
        .with_build_side(JoinSide::Right),
        Arc::clone(&joined_schema),
    );
    let presence = builder.add_projection(
        joined_b,
        projection_outputs(vec![Expr::literal(LiteralValue::Int32(1)).alias("__row_present")], presence_schema),
    );
    let count = builder.add_aggregate(
        presence,
        AggregateSpec::new(
            Vec::new(),
            vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("__row_present"), "cnt")],
        ),
        Arc::clone(&count_schema),
    );
    let projection = builder.add_projection(count, projection_outputs(vec![Expr::column("cnt")], count_schema));
    builder.finish_with_host_sink(projection)
}

#[allow(dead_code)]
pub(super) fn projection_root_keyed_grouped_aggregate_input_plan() -> QueryPlan {
    let fact_schema = Arc::new(Schema::new(vec![
        Field::new("fact_k", DataType::Int32, false),
        Field::new("dim_a", DataType::Int32, false),
        Field::new("dim_b", DataType::Int32, false),
    ]));
    let dim_schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false)]));
    let joined_schema = Arc::new(Schema::new(vec![Field::new("fact_k", DataType::Int32, false)]));
    let grouped_schema = Arc::new(Schema::new(vec![
        Field::new("fact_k", DataType::Int32, false),
        Field::new("cnt", DataType::Int64, false),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let fact = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/projection-keyed-grouped-aggregate-input-fact.parquet",
        fact_schema,
        vec!["fact_k".into(), "dim_a".into(), "dim_b".into()],
    )));
    let dim_a = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/projection-keyed-grouped-aggregate-input-dim-a.parquet",
        Arc::clone(&dim_schema),
        vec!["k".into()],
    )));
    let dim_b = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/projection-keyed-grouped-aggregate-input-dim-b.parquet",
        Arc::clone(&dim_schema),
        vec!["k".into()],
    )));
    let joined_a = builder.add_join(
        fact,
        dim_a,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "dim_a", "dim_a"), ColumnRef::input(1, "k", "k"))],
        )
        .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "fact_k", "fact_k"), "fact_k")])
        .with_build_side(JoinSide::Right),
        Arc::clone(&joined_schema),
    );
    let joined_b = builder.add_join(
        joined_a,
        dim_b,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "fact_k", "fact_k"), ColumnRef::input(1, "k", "k"))],
        )
        .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "fact_k", "fact_k"), "fact_k")])
        .with_build_side(JoinSide::Right),
        Arc::clone(&joined_schema),
    );
    let aggregate = builder.add_aggregate(
        joined_b,
        AggregateSpec::new(
            vec![Expr::column("fact_k")],
            vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("fact_k"), "cnt")],
        ),
        Arc::clone(&grouped_schema),
    );
    let projection = builder.add_projection(
        aggregate,
        projection_outputs(vec![Expr::column("fact_k"), Expr::column("cnt")], grouped_schema),
    );
    builder.finish_with_host_sink(projection)
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProjectionScalarSetopPlanVariant {
    ScalarFanout,
    OneRowDriver,
    ScanLimitDriver,
    MaxRowsDriver,
    LimitDriver,
    UnsupportedJoin,
    SetopDedup,
    SetopDedupInput,
    NonDedupLeftAntiInput,
    DistinctAggregate,
    RepeatedScalarFanout,
    RepeatedScalarFanoutUnboundedInput,
}

#[allow(dead_code)]
fn projection_root_scalar_fanout_plan_with_variant(variant: ProjectionScalarSetopPlanVariant) -> QueryPlan {
    let driver_schema = Arc::new(Schema::new(vec![Field::new("driver_id", DataType::Int32, false)]));
    let fact_schema = Arc::new(Schema::new(vec![
        Field::new("amount", DataType::Int64, true),
        Field::new("qty", DataType::Int32, false),
    ]));
    let driver_projection_schema = Arc::new(Schema::new(Vec::<Field>::new()));
    let count_arg_schema = Arc::new(Schema::new(vec![Field::new("qty", DataType::Int32, false)]));
    let count_schema = Arc::new(Schema::new(vec![Field::new("cnt", DataType::Int64, false)]));
    let avg_schema = Arc::new(Schema::new(vec![Field::new("avg_amount", DataType::Float64, true)]));
    let join_schema = Arc::new(Schema::new(vec![
        Field::new("cnt", DataType::Int64, false),
        Field::new("avg_amount", DataType::Float64, true),
    ]));
    let output_schema = Arc::new(Schema::new(vec![Field::new("bucket", DataType::Float64, true)]));

    let mut builder = QueryPlanBuilder::new();
    let driver = match variant {
        ProjectionScalarSetopPlanVariant::OneRowDriver => {
            builder.add_source(SourceNode::LiteralBatch(nexus_query_engine::LiteralBatchSource::placeholder_row()))
        },
        ProjectionScalarSetopPlanVariant::ScanLimitDriver => {
            let driver = builder.add_source(SourceNode::LocalParquet(
                LocalParquetSource::new(
                    "/tmp/projection-scalar-setop-scan-limit-driver.parquet",
                    driver_schema,
                    vec!["driver_id".into()],
                )
                .with_scan_limit(Some(1)),
            ));
            builder.add_projection(driver, projection_outputs(Vec::new(), driver_projection_schema))
        },
        ProjectionScalarSetopPlanVariant::MaxRowsDriver => {
            let driver = builder.add_source(SourceNode::LocalParquet(
                LocalParquetSource::new(
                    "/tmp/projection-scalar-setop-max-rows-driver.parquet",
                    driver_schema,
                    vec!["driver_id".into()],
                )
                .with_max_rows(Some(1)),
            ));
            builder.add_projection(driver, projection_outputs(Vec::new(), driver_projection_schema))
        },
        ProjectionScalarSetopPlanVariant::LimitDriver => {
            let driver = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
                "/tmp/projection-scalar-setop-limit-driver.parquet",
                Arc::clone(&driver_schema),
                vec!["driver_id".into()],
            )));
            let driver = builder.add_limit(driver, LimitSpec::fetch(1), driver_schema);
            builder.add_projection(driver, projection_outputs(Vec::new(), driver_projection_schema))
        },
        ProjectionScalarSetopPlanVariant::ScalarFanout
        | ProjectionScalarSetopPlanVariant::UnsupportedJoin
        | ProjectionScalarSetopPlanVariant::SetopDedup
        | ProjectionScalarSetopPlanVariant::SetopDedupInput
        | ProjectionScalarSetopPlanVariant::NonDedupLeftAntiInput
        | ProjectionScalarSetopPlanVariant::DistinctAggregate
        | ProjectionScalarSetopPlanVariant::RepeatedScalarFanout
        | ProjectionScalarSetopPlanVariant::RepeatedScalarFanoutUnboundedInput => {
            let driver = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
                "/tmp/projection-scalar-setop-driver.parquet",
                driver_schema,
                vec!["driver_id".into()],
            )));
            let driver = builder.add_filter(
                driver,
                Expr::binary(Expr::column("driver_id"), BinaryOp::Eq, Expr::literal(LiteralValue::Int32(1))),
                Arc::new(Schema::new(vec![Field::new("driver_id", DataType::Int32, false)])),
            );
            builder.add_projection(driver, projection_outputs(Vec::new(), driver_projection_schema))
        },
    };
    let fact = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/projection-scalar-setop-fact.parquet",
        fact_schema,
        vec!["amount".into(), "qty".into()],
    )));
    let count_projection =
        builder.add_projection(fact, projection_outputs(vec![Expr::column("qty")], Arc::clone(&count_arg_schema)));
    let count = builder.add_aggregate(
        count_projection,
        AggregateSpec::new(Vec::new(), vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("qty"), "cnt")]),
        Arc::clone(&count_schema),
    );
    let avg_projection = builder.add_projection(
        fact,
        projection_outputs(
            vec![Expr::column("amount")],
            Arc::new(Schema::new(vec![Field::new("amount", DataType::Int64, true)])),
        ),
    );
    let avg = builder.add_aggregate(
        avg_projection,
        AggregateSpec::new(
            Vec::new(),
            vec![AggregateExpr::new(AggregateFunction::Avg, Expr::column("amount"), "avg_amount")],
        ),
        Arc::clone(&avg_schema),
    );
    let scalar_join = builder.add_join(
        count,
        avg,
        JoinSpec::new(JoinType::Cross, Vec::new())
            .with_output_projection(vec![
                JoinOutputColumn::left(ColumnRef::input(0, "cnt", "cnt"), "cnt"),
                JoinOutputColumn::right(ColumnRef::input(1, "avg_amount", "avg_amount"), "avg_amount"),
            ])
            .with_build_side(JoinSide::Left),
        Arc::clone(&join_schema),
    );
    let join_type =
        if variant == ProjectionScalarSetopPlanVariant::UnsupportedJoin { JoinType::LeftAnti } else { JoinType::Left };
    let root_join = builder.add_join(
        driver,
        scalar_join,
        JoinSpec::new(join_type, Vec::new())
            .with_output_projection(vec![
                JoinOutputColumn::right(ColumnRef::input(1, "cnt", "cnt"), "cnt"),
                JoinOutputColumn::right(ColumnRef::input(1, "avg_amount", "avg_amount"), "avg_amount"),
            ])
            .with_build_side(JoinSide::Right),
        Arc::clone(&join_schema),
    );
    let projection = builder.add_projection(
        root_join,
        projection_outputs(
            vec![Expr::Case {
                branches: vec![CaseWhen::new(
                    Expr::binary(Expr::column("cnt"), BinaryOp::Gt, Expr::literal(LiteralValue::Int64(0))),
                    Expr::column("avg_amount"),
                )],
                else_expr: Box::new(Expr::literal(LiteralValue::Float64(0.0))),
            }],
            output_schema,
        ),
    );
    builder.finish_with_host_sink(projection)
}

#[allow(dead_code)]
fn projection_root_repeated_scalar_fanout_plan_with_variant(variant: ProjectionScalarSetopPlanVariant) -> QueryPlan {
    let driver_schema = Arc::new(Schema::new(vec![Field::new("driver_id", DataType::Int32, false)]));
    let fact_schema = Arc::new(Schema::new(vec![
        Field::new("amount", DataType::Int64, true),
        Field::new("qty", DataType::Int32, false),
    ]));
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("count_all", DataType::Int64, false),
        Field::new("sum_qty", DataType::Int64, true),
        Field::new("avg_amount", DataType::Float64, true),
        Field::new("max_qty", DataType::Int32, true),
    ]));
    let count_schema = Arc::new(Schema::new(vec![Field::new("count_all", DataType::Int64, false)]));
    let sum_schema = Arc::new(Schema::new(vec![Field::new("sum_qty", DataType::Int64, true)]));
    let avg_schema = Arc::new(Schema::new(vec![Field::new("avg_amount", DataType::Float64, true)]));
    let max_schema = Arc::new(Schema::new(vec![Field::new("max_qty", DataType::Int32, true)]));
    let count_sum_schema = Arc::new(Schema::new(vec![
        Field::new("count_all", DataType::Int64, false),
        Field::new("sum_qty", DataType::Int64, true),
    ]));
    let count_sum_avg_schema = Arc::new(Schema::new(vec![
        Field::new("count_all", DataType::Int64, false),
        Field::new("sum_qty", DataType::Int64, true),
        Field::new("avg_amount", DataType::Float64, true),
    ]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("count_all", DataType::Int64, false),
        Field::new("sum_qty", DataType::Int64, true),
        Field::new("avg_amount", DataType::Float64, true),
        Field::new("max_qty", DataType::Int32, true),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let fact = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/projection-repeated-scalar-setop-fact.parquet",
        fact_schema,
        vec!["amount".into(), "qty".into()],
    )));
    let aggregate = builder.add_aggregate(
        fact,
        AggregateSpec::new(
            Vec::new(),
            vec![
                AggregateExpr::new(AggregateFunction::Count, Expr::column("qty"), "count_all"),
                AggregateExpr::new(AggregateFunction::Sum, Expr::column("qty"), "sum_qty"),
                AggregateExpr::new(AggregateFunction::Avg, Expr::column("amount"), "avg_amount"),
                AggregateExpr::new(AggregateFunction::Max, Expr::column("qty"), "max_qty"),
            ],
        ),
        aggregate_schema,
    );
    let count = builder
        .add_projection(aggregate, projection_outputs(vec![Expr::column("count_all")], Arc::clone(&count_schema)));
    let sum =
        builder.add_projection(aggregate, projection_outputs(vec![Expr::column("sum_qty")], Arc::clone(&sum_schema)));
    let avg = builder
        .add_projection(aggregate, projection_outputs(vec![Expr::column("avg_amount")], Arc::clone(&avg_schema)));
    let max =
        builder.add_projection(aggregate, projection_outputs(vec![Expr::column("max_qty")], Arc::clone(&max_schema)));
    let count_sum = builder.add_join(
        count,
        sum,
        JoinSpec::new(JoinType::Cross, Vec::new())
            .with_output_projection(vec![
                JoinOutputColumn::left(ColumnRef::input(0, "count_all", "count_all"), "count_all"),
                JoinOutputColumn::right(ColumnRef::input(1, "sum_qty", "sum_qty"), "sum_qty"),
            ])
            .with_build_side(JoinSide::Left),
        Arc::clone(&count_sum_schema),
    );
    let count_sum_avg = builder.add_join(
        count_sum,
        avg,
        JoinSpec::new(JoinType::Cross, Vec::new())
            .with_output_projection(vec![
                JoinOutputColumn::left(ColumnRef::input(0, "count_all", "count_all"), "count_all"),
                JoinOutputColumn::left(ColumnRef::input(0, "sum_qty", "sum_qty"), "sum_qty"),
                JoinOutputColumn::right(ColumnRef::input(1, "avg_amount", "avg_amount"), "avg_amount"),
            ])
            .with_build_side(JoinSide::Right),
        Arc::clone(&count_sum_avg_schema),
    );
    let repeated = builder.add_join(
        count_sum_avg,
        max,
        JoinSpec::new(JoinType::Cross, Vec::new())
            .with_output_projection(vec![
                JoinOutputColumn::left(ColumnRef::input(0, "count_all", "count_all"), "count_all"),
                JoinOutputColumn::left(ColumnRef::input(0, "sum_qty", "sum_qty"), "sum_qty"),
                JoinOutputColumn::left(ColumnRef::input(0, "avg_amount", "avg_amount"), "avg_amount"),
                JoinOutputColumn::right(ColumnRef::input(1, "max_qty", "max_qty"), "max_qty"),
            ])
            .with_build_side(JoinSide::Right),
        Arc::clone(&output_schema),
    );
    let root_input = if variant == ProjectionScalarSetopPlanVariant::RepeatedScalarFanoutUnboundedInput {
        let driver = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
            "/tmp/projection-repeated-scalar-setop-driver.parquet",
            driver_schema,
            vec!["driver_id".into()],
        )));
        builder.add_join(
            driver,
            repeated,
            JoinSpec::new(JoinType::Cross, Vec::new())
                .with_output_projection(vec![
                    JoinOutputColumn::right(ColumnRef::input(1, "count_all", "count_all"), "count_all"),
                    JoinOutputColumn::right(ColumnRef::input(1, "sum_qty", "sum_qty"), "sum_qty"),
                    JoinOutputColumn::right(ColumnRef::input(1, "avg_amount", "avg_amount"), "avg_amount"),
                    JoinOutputColumn::right(ColumnRef::input(1, "max_qty", "max_qty"), "max_qty"),
                ])
                .with_build_side(JoinSide::Right),
            Arc::clone(&output_schema),
        )
    } else {
        repeated
    };
    let projection = builder.add_projection(
        root_input,
        projection_outputs(
            vec![
                Expr::column("count_all"),
                Expr::column("sum_qty"),
                Expr::column("avg_amount"),
                Expr::column("max_qty"),
            ],
            output_schema,
        ),
    );
    builder.finish_with_host_sink(projection)
}

#[allow(dead_code)]
fn projection_root_setop_dedup_scalar_plan_with_variant(variant: ProjectionScalarSetopPlanVariant) -> QueryPlan {
    let left_schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false)]));
    let right_schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false)]));
    let dedup_schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false)]));
    let one_schema = Arc::new(Schema::new(vec![Field::new("one", DataType::Int64, false)]));
    let scalar_schema = Arc::new(Schema::new(vec![Field::new("count", DataType::Int64, false)]));

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/projection-setop-left.parquet",
        Arc::clone(&left_schema),
        vec!["k".into()],
    )));
    let right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/projection-setop-right.parquet",
        right_schema,
        vec!["k".into()],
    )));
    let left_dedup = if variant == ProjectionScalarSetopPlanVariant::NonDedupLeftAntiInput {
        left
    } else {
        builder.add_aggregate(left, AggregateSpec::new(vec![Expr::column("k")], Vec::new()), Arc::clone(&dedup_schema))
    };
    let (right_aggregate_exprs, right_dedup_schema) = if variant == ProjectionScalarSetopPlanVariant::DistinctAggregate
    {
        (
            vec![AggregateExpr::new(AggregateFunction::NUnique, Expr::column("k"), "nunique_k")],
            Arc::new(Schema::new(vec![
                Field::new("k", DataType::Int32, false),
                Field::new("nunique_k", DataType::UInt64, false),
            ])),
        )
    } else {
        (Vec::new(), Arc::clone(&dedup_schema))
    };
    let right_dedup = builder.add_aggregate(
        right,
        AggregateSpec::new(vec![Expr::column("k")], right_aggregate_exprs),
        right_dedup_schema,
    );
    let anti = builder.add_join(
        left_dedup,
        right_dedup,
        JoinSpec::new(
            JoinType::LeftAnti,
            vec![JoinKey::new(ColumnRef::input(0, "k", "k"), ColumnRef::input(1, "k", "k"))],
        )
        .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "k", "k"), "k")])
        .with_build_side(JoinSide::Right),
        Arc::clone(&dedup_schema),
    );
    if variant == ProjectionScalarSetopPlanVariant::SetopDedupInput {
        let projection = builder.add_projection(anti, projection_outputs(vec![Expr::column("k")], dedup_schema));
        return builder.finish_with_host_sink(projection);
    }
    let count_input = builder
        .add_projection(anti, projection_outputs(vec![Expr::literal(LiteralValue::Int64(1)).alias("one")], one_schema));
    let count = builder.add_aggregate(
        count_input,
        AggregateSpec::new(
            Vec::new(),
            vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("one"), "count")],
        ),
        Arc::clone(&scalar_schema),
    );
    let projection = builder.add_projection(count, projection_outputs(vec![Expr::column("count")], scalar_schema));
    builder.finish_with_host_sink(projection)
}

pub(super) fn projection_root_global_limit_distinct_join_aggregate_plan() -> QueryPlan {
    projection_root_global_limit_join_aggregate_plan_with_variant(ProjectionGlobalLimitPlanVariant::Distinct)
}

pub(super) fn projection_root_global_limit_count_distinct_join_aggregate_plan() -> QueryPlan {
    projection_root_global_limit_join_aggregate_plan_with_variant(ProjectionGlobalLimitPlanVariant::CountDistinct)
}

pub(super) fn projection_root_global_limit_dedup_join_aggregate_plan() -> QueryPlan {
    projection_root_global_limit_join_aggregate_plan_with_variant(ProjectionGlobalLimitPlanVariant::Dedup)
}

pub(super) fn projection_root_global_limit_filtered_distinct_join_aggregate_plan() -> QueryPlan {
    projection_root_global_limit_join_aggregate_plan_with_variant(ProjectionGlobalLimitPlanVariant::FilteredDistinct)
}

pub(super) fn projection_root_global_limit_non_count_distinct_join_aggregate_plan() -> QueryPlan {
    projection_root_global_limit_join_aggregate_plan_with_variant(ProjectionGlobalLimitPlanVariant::NonCountDistinct)
}

pub(super) fn projection_root_no_fetch_limit_join_aggregate_plan() -> QueryPlan {
    projection_root_global_limit_join_aggregate_plan_with_variant(ProjectionGlobalLimitPlanVariant::NoFetch)
}

pub(super) fn projection_root_global_limit_join_without_aggregate_plan() -> QueryPlan {
    projection_root_global_limit_join_aggregate_plan_with_variant(ProjectionGlobalLimitPlanVariant::NoAggregate)
}

pub(super) fn projection_root_global_limit_aggregate_without_join_plan() -> QueryPlan {
    projection_root_global_limit_join_aggregate_plan_with_variant(ProjectionGlobalLimitPlanVariant::NoJoin)
}

pub(super) fn projection_root_global_limit_over_sort_join_aggregate_plan() -> QueryPlan {
    projection_root_global_limit_join_aggregate_plan_with_variant(ProjectionGlobalLimitPlanVariant::SortLimitInput)
}

pub(super) fn projection_root_global_limit_over_filter_sort_join_aggregate_plan() -> QueryPlan {
    projection_root_global_limit_join_aggregate_plan_with_variant(
        ProjectionGlobalLimitPlanVariant::FilterSortLimitInput,
    )
}

pub(super) fn projection_root_filter_after_global_limit_join_aggregate_plan() -> QueryPlan {
    projection_root_global_limit_join_aggregate_plan_with_variant(ProjectionGlobalLimitPlanVariant::PostLimitFilter)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProjectionGlobalLimitPlanVariant {
    Standard,
    Distinct,
    CountDistinct,
    Dedup,
    FilteredDistinct,
    NonCountDistinct,
    NoFetch,
    NoAggregate,
    NoJoin,
    SortLimitInput,
    FilterSortLimitInput,
    PostLimitFilter,
}

fn projection_root_global_limit_join_aggregate_plan_with_variant(
    variant: ProjectionGlobalLimitPlanVariant,
) -> QueryPlan {
    let left_schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("d", DataType::Int32, false),
        Field::new("v", DataType::Int32, true),
    ]));
    let right_schema = Arc::new(Schema::new(vec![Field::new("d", DataType::Int32, false)]));
    let join_schema =
        Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false), Field::new("v", DataType::Int32, true)]));
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("metric", DataType::Int64, false),
    ]));
    let dedup_schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false)]));
    let metric_projection_schema = Arc::new(Schema::new(vec![
        Field::new("metric", DataType::Int64, false),
        Field::new("k", DataType::Int32, false),
    ]));
    let dedup_projection_schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false)]));
    let join_projection_schema =
        Arc::new(Schema::new(vec![Field::new("v", DataType::Int32, true), Field::new("k", DataType::Int32, false)]));

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/projection-global-limit-left.parquet",
        Arc::clone(&left_schema),
        vec!["k".into(), "d".into(), "v".into()],
    )));
    let joined_input = if variant == ProjectionGlobalLimitPlanVariant::NoJoin {
        left
    } else {
        let right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
            "/tmp/projection-global-limit-right.parquet",
            right_schema,
            vec!["d".into()],
        )));
        builder.add_join(
            left,
            right,
            JoinSpec::new(
                JoinType::Inner,
                vec![JoinKey::new(ColumnRef::input(0, "d", "d"), ColumnRef::input(1, "d", "d"))],
            )
            .with_output_projection(vec![
                JoinOutputColumn::left(ColumnRef::input(0, "k", "k"), "k"),
                JoinOutputColumn::left(ColumnRef::input(0, "v", "v"), "v"),
            ])
            .with_build_side(JoinSide::Right),
            Arc::clone(&join_schema),
        )
    };

    let (limit_input, limit_schema, projection_exprs, projection_schema) =
        if variant == ProjectionGlobalLimitPlanVariant::NoAggregate {
            (
                joined_input,
                Arc::clone(&join_schema),
                vec![Expr::column("v"), Expr::column("k")],
                Arc::clone(&join_projection_schema),
            )
        } else if variant == ProjectionGlobalLimitPlanVariant::Dedup {
            let aggregate = builder.add_aggregate(
                joined_input,
                AggregateSpec::new(vec![Expr::column("k")], Vec::new()),
                Arc::clone(&dedup_schema),
            );
            (aggregate, Arc::clone(&dedup_schema), vec![Expr::column("k")], Arc::clone(&dedup_projection_schema))
        } else {
            let aggregate_expr = match variant {
                ProjectionGlobalLimitPlanVariant::Distinct => {
                    AggregateExpr::new(AggregateFunction::NUnique, Expr::column("v"), "metric")
                },
                ProjectionGlobalLimitPlanVariant::CountDistinct => {
                    AggregateExpr::new(AggregateFunction::Count, Expr::column("v"), "metric").with_distinct(true)
                },
                ProjectionGlobalLimitPlanVariant::FilteredDistinct => {
                    AggregateExpr::new(AggregateFunction::NUnique, Expr::column("v"), "metric").with_filter(Some(
                        Expr::binary(Expr::column("v"), BinaryOp::GtEq, Expr::literal(LiteralValue::Int32(0))),
                    ))
                },
                ProjectionGlobalLimitPlanVariant::NonCountDistinct => {
                    AggregateExpr::new(AggregateFunction::Sum, Expr::column("v"), "metric").with_distinct(true)
                },
                _ => AggregateExpr::new(AggregateFunction::Count, Expr::column("v"), "metric"),
            };
            let aggregate = builder.add_aggregate(
                joined_input,
                AggregateSpec::new(vec![Expr::column("k")], vec![aggregate_expr]),
                Arc::clone(&aggregate_schema),
            );
            (
                aggregate,
                Arc::clone(&aggregate_schema),
                vec![Expr::column("metric"), Expr::column("k")],
                Arc::clone(&metric_projection_schema),
            )
        };
    let bounded_input = if matches!(
        variant,
        ProjectionGlobalLimitPlanVariant::SortLimitInput | ProjectionGlobalLimitPlanVariant::FilterSortLimitInput
    ) {
        let sort = builder.add_sort(
            limit_input,
            SortSpec::new(vec![SortExpr::new(Expr::column("metric"), true, false)]),
            Arc::clone(&limit_schema),
        );
        if variant == ProjectionGlobalLimitPlanVariant::FilterSortLimitInput {
            builder.add_filter(
                sort,
                Expr::binary(Expr::column("metric"), BinaryOp::GtEq, Expr::literal(LiteralValue::Int64(0))),
                Arc::clone(&limit_schema),
            )
        } else {
            sort
        }
    } else {
        limit_input
    };
    let limit_spec =
        if variant == ProjectionGlobalLimitPlanVariant::NoFetch { LimitSpec::offset(0) } else { LimitSpec::fetch(10) };
    let limit = builder.add_limit(bounded_input, limit_spec, Arc::clone(&limit_schema));
    let projection_input = if variant == ProjectionGlobalLimitPlanVariant::PostLimitFilter {
        builder.add_filter(
            limit,
            Expr::binary(Expr::column("metric"), BinaryOp::GtEq, Expr::literal(LiteralValue::Int64(0))),
            Arc::clone(&limit_schema),
        )
    } else {
        limit
    };
    let projection = builder.add_projection(projection_input, projection_outputs(projection_exprs, projection_schema));
    builder.finish_with_host_sink(projection)
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
        AggregateFunction::StddevSamp
        | AggregateFunction::StddevPop
        | AggregateFunction::VarSamp
        | AggregateFunction::VarPop => DataType::Float64,
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
