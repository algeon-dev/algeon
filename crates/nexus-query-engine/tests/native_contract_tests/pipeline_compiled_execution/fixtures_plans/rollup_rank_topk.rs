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

pub(super) fn selected_q22_class_rollup_topk_after_pushdown_plan() -> QueryPlan {
    let fact_schema = Arc::new(Schema::new(vec![
        Field::new("date_sk", DataType::Int32, false),
        Field::new("item_sk", DataType::Int32, false),
        Field::new("quantity_on_hand", DataType::Int64, true),
    ]));
    let date_schema = Arc::new(Schema::new(vec![Field::new("date_sk", DataType::Int32, false)]));
    let item_schema = Arc::new(Schema::new(vec![
        Field::new("item_sk", DataType::Int32, false),
        Field::new("product_name", DataType::Utf8, true),
        Field::new("brand", DataType::Utf8, true),
        Field::new("class", DataType::Utf8, true),
        Field::new("category", DataType::Utf8, true),
    ]));
    let fact_date_join_schema = Arc::new(Schema::new(vec![
        Field::new("item_sk", DataType::Int32, false),
        Field::new("quantity_on_hand", DataType::Int64, true),
    ]));
    let item_join_schema = Arc::new(Schema::new(vec![
        Field::new("product_name", DataType::Utf8, true),
        Field::new("brand", DataType::Utf8, true),
        Field::new("class", DataType::Utf8, true),
        Field::new("category", DataType::Utf8, true),
        Field::new("quantity_on_hand", DataType::Int64, true),
    ]));
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("product_name", DataType::Utf8, true),
        Field::new("brand", DataType::Utf8, true),
        Field::new("class", DataType::Utf8, true),
        Field::new("category", DataType::Utf8, true),
        Field::new("__grouping_id", DataType::UInt32, false),
        Field::new("avg_qoh", DataType::Float64, true),
    ]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("product_name", DataType::Utf8, true),
        Field::new("brand", DataType::Utf8, true),
        Field::new("class", DataType::Utf8, true),
        Field::new("category", DataType::Utf8, true),
        Field::new("qoh", DataType::Float64, true),
    ]));

    let fact_file = LocalParquetFile::new("/tmp/compiled-selected-q22-class-rollup-topk-fact.parquet")
        .with_file_size_bytes(512 << 20)
        .with_row_groups(vec![0, 1])
        .with_projected_row_group_bytes(vec![32 << 20, 32 << 20, 32 << 20, 32 << 20], vec![32 << 20; 4])
        .with_row_group_rows(vec![1_000_000; 4]);
    let date_file =
        LocalParquetFile::new("/tmp/compiled-selected-q22-class-rollup-topk-date.parquet").with_file_size_bytes(1024);
    let item_file =
        LocalParquetFile::new("/tmp/compiled-selected-q22-class-rollup-topk-item.parquet").with_file_size_bytes(4096);

    let mut builder = QueryPlanBuilder::new();
    let fact = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![fact_file],
        fact_schema,
        vec!["date_sk".into(), "item_sk".into(), "quantity_on_hand".into()],
    )));
    let date = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![date_file],
        date_schema,
        vec!["date_sk".into()],
    )));
    let fact_date_join = builder.add_join(
        fact,
        date,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "date_sk", "date_sk"), ColumnRef::input(1, "date_sk", "date_sk"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "item_sk", "item_sk"), "item_sk"),
            JoinOutputColumn::left(ColumnRef::input(0, "quantity_on_hand", "quantity_on_hand"), "quantity_on_hand"),
        ])
        .with_build_side(JoinSide::Right),
        fact_date_join_schema,
    );
    let item = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![item_file],
        item_schema,
        vec!["item_sk".into(), "product_name".into(), "brand".into(), "class".into(), "category".into()],
    )));
    let item_join = builder.add_join(
        fact_date_join,
        item,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "item_sk", "item_sk"), ColumnRef::input(1, "item_sk", "item_sk"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::right(ColumnRef::input(1, "product_name", "product_name"), "product_name"),
            JoinOutputColumn::right(ColumnRef::input(1, "brand", "brand"), "brand"),
            JoinOutputColumn::right(ColumnRef::input(1, "class", "class"), "class"),
            JoinOutputColumn::right(ColumnRef::input(1, "category", "category"), "category"),
            JoinOutputColumn::left(ColumnRef::input(0, "quantity_on_hand", "quantity_on_hand"), "quantity_on_hand"),
        ])
        .with_build_side(JoinSide::Right),
        item_join_schema,
    );
    let aggregate = builder.add_aggregate(
        item_join,
        AggregateSpec::new(
            vec![Expr::column("product_name"), Expr::column("brand"), Expr::column("class"), Expr::column("category")],
            vec![AggregateExpr::new(AggregateFunction::Avg, Expr::column("quantity_on_hand"), "avg_qoh")],
        )
        .with_grouping_sets(GroupingSetsSpec::new(
            vec![
                vec![true, true, true, true],
                vec![false, true, true, true],
                vec![false, false, true, true],
                vec![false, false, false, true],
                vec![false, false, false, false],
            ],
            "__grouping_id",
        )),
        aggregate_schema,
    );
    let projection = builder.add_projection(
        aggregate,
        vec![
            ProjectionOutput::new(Expr::column("product_name"), output_schema.field(0).as_ref().clone()),
            ProjectionOutput::new(Expr::column("brand"), output_schema.field(1).as_ref().clone()),
            ProjectionOutput::new(Expr::column("class"), output_schema.field(2).as_ref().clone()),
            ProjectionOutput::new(Expr::column("category"), output_schema.field(3).as_ref().clone()),
            ProjectionOutput::new(Expr::column("avg_qoh").alias("qoh"), output_schema.field(4).as_ref().clone()),
        ],
    );
    let sort = builder.add_sort(
        projection,
        SortSpec::new(vec![
            SortExpr::new(Expr::column("qoh"), false, false),
            SortExpr::new(Expr::column("product_name"), false, false),
            SortExpr::new(Expr::column("brand"), false, false),
            SortExpr::new(Expr::column("class"), false, false),
            SortExpr::new(Expr::column("category"), false, false),
        ]),
        Arc::clone(&output_schema),
    );
    let limit = builder.add_limit(sort, LimitSpec::fetch(100), output_schema);
    builder.finish_with_host_sink(limit)
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

#[expect(clippy::too_many_arguments, reason = "test fixture parameters are independent fixture axes")]
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
                .alias("__nexus_native_aggregate_arg_0"),
                Field::new("__nexus_native_aggregate_arg_0", DataType::Int64, false),
            ),
        ],
    );
    let aggregate = builder.add_aggregate(
        coalesce_projection,
        AggregateSpec::new(
            vec![Expr::column("left_group"), Expr::column("right_group")],
            vec![AggregateExpr::new(
                AggregateFunction::Sum,
                Expr::column("__nexus_native_aggregate_arg_0"),
                "sumsales",
            )],
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

#[expect(clippy::too_many_arguments, reason = "test fixture parameters are independent fixture axes")]
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
