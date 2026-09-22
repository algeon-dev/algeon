pub(super) fn projection_root_scalar_fanout_plan() -> QueryPlan {
    projection_root_scalar_fanout_plan_with_variant(ProjectionScalarSetopPlanVariant::ScalarFanout)
}

pub(super) fn projection_root_scalar_fanout_one_row_driver_plan() -> QueryPlan {
    projection_root_scalar_fanout_plan_with_variant(ProjectionScalarSetopPlanVariant::OneRowDriver)
}

pub(super) fn projection_root_scalar_fanout_scan_limit_driver_plan() -> QueryPlan {
    projection_root_scalar_fanout_plan_with_variant(ProjectionScalarSetopPlanVariant::ScanLimitDriver)
}

pub(super) fn projection_root_scalar_fanout_max_rows_driver_plan() -> QueryPlan {
    projection_root_scalar_fanout_plan_with_variant(ProjectionScalarSetopPlanVariant::MaxRowsDriver)
}

pub(super) fn projection_root_scalar_fanout_limit_driver_plan() -> QueryPlan {
    projection_root_scalar_fanout_plan_with_variant(ProjectionScalarSetopPlanVariant::LimitDriver)
}

pub(super) fn projection_root_scalar_fanout_unsupported_join_plan() -> QueryPlan {
    projection_root_scalar_fanout_plan_with_variant(ProjectionScalarSetopPlanVariant::UnsupportedJoin)
}

pub(super) fn projection_root_repeated_scalar_fanout_plan() -> QueryPlan {
    projection_root_repeated_scalar_fanout_plan_with_variant(ProjectionScalarSetopPlanVariant::RepeatedScalarFanout)
}

pub(super) fn projection_root_repeated_scalar_fanout_unbounded_input_plan() -> QueryPlan {
    projection_root_repeated_scalar_fanout_plan_with_variant(
        ProjectionScalarSetopPlanVariant::RepeatedScalarFanoutUnboundedInput,
    )
}

pub(super) fn projection_root_setop_dedup_scalar_plan() -> QueryPlan {
    projection_root_setop_dedup_scalar_plan_with_variant(ProjectionScalarSetopPlanVariant::SetopDedup)
}

pub(super) fn projection_root_setop_dedup_input_plan() -> QueryPlan {
    projection_root_setop_dedup_scalar_plan_with_variant(ProjectionScalarSetopPlanVariant::SetopDedupInput)
}

pub(super) fn projection_root_setop_dedup_scalar_non_dedup_left_anti_input_plan() -> QueryPlan {
    projection_root_setop_dedup_scalar_plan_with_variant(ProjectionScalarSetopPlanVariant::NonDedupLeftAntiInput)
}

pub(super) fn projection_root_setop_distinct_aggregate_plan() -> QueryPlan {
    projection_root_setop_dedup_scalar_plan_with_variant(ProjectionScalarSetopPlanVariant::DistinctAggregate)
}

pub(super) fn projection_root_left_anti_scalar_count_plan() -> QueryPlan {
    let left_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("value", DataType::Utf8, true),
    ]));
    let right_schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
    let one_schema = Arc::new(Schema::new(vec![Field::new("one", DataType::Int64, false)]));
    let count_schema = Arc::new(Schema::new(vec![Field::new("count", DataType::Int64, false)]));

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/projection-left-anti-count-left.parquet",
        Arc::clone(&left_schema),
        vec!["id".into(), "value".into()],
    )));
    let right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/projection-left-anti-count-right.parquet",
        right_schema,
        vec!["id".into()],
    )));
    let anti = builder.add_join(
        left,
        right,
        JoinSpec::new(
            JoinType::LeftAnti,
            vec![JoinKey::new(ColumnRef::input(0, "id", "id"), ColumnRef::input(1, "id", "id"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "id", "id"), "id"),
            JoinOutputColumn::left(ColumnRef::input(0, "value", "value"), "value"),
        ])
        .with_build_side(JoinSide::Right),
        left_schema,
    );
    let count_input = builder
        .add_projection(anti, projection_outputs(vec![Expr::literal(LiteralValue::Int64(1)).alias("one")], one_schema));
    let count = builder.add_aggregate(
        count_input,
        AggregateSpec::new(
            Vec::new(),
            vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("one"), "count")],
        ),
        Arc::clone(&count_schema),
    );
    let projection = builder.add_projection(count, projection_outputs(vec![Expr::column("count")], count_schema));
    builder.finish_with_host_sink(projection)
}

pub(super) fn projection_root_scalar_join_grouped_avg_input_plan() -> QueryPlan {
    let left_schema = Arc::new(Schema::new(vec![
        Field::new("vendor", DataType::Int32, true),
        Field::new("score", DataType::Float64, true),
    ]));
    let grouped_schema = Arc::new(Schema::new(vec![
        Field::new("vendor", DataType::Int32, true),
        Field::new("avg_score", DataType::Float64, true),
    ]));
    let join_schema = Arc::new(Schema::new(vec![
        Field::new("score", DataType::Float64, true),
        Field::new("avg_score", DataType::Float64, true),
    ]));
    let one_schema = Arc::new(Schema::new(vec![Field::new("one", DataType::Int64, false)]));
    let count_schema = Arc::new(Schema::new(vec![Field::new("count", DataType::Int64, false)]));

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/projection-scalar-join-grouped-avg-left.parquet",
        Arc::clone(&left_schema),
        vec!["vendor".into(), "score".into()],
    )));
    let right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/projection-scalar-join-grouped-avg-right.parquet",
        Arc::clone(&left_schema),
        vec!["vendor".into(), "score".into()],
    )));
    let grouped = builder.add_aggregate(
        right,
        AggregateSpec::new(
            vec![Expr::column("vendor")],
            vec![AggregateExpr::new(AggregateFunction::Avg, Expr::column("score"), "avg_score")],
        ),
        Arc::clone(&grouped_schema),
    );
    let join = builder.add_join(
        left,
        grouped,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "vendor", "left_vendor"),
                ColumnRef::input(1, "vendor", "right_vendor"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "score", "score"), "score"),
            JoinOutputColumn::right(ColumnRef::input(1, "avg_score", "avg_score"), "avg_score"),
        ])
        .with_build_side(JoinSide::Right),
        Arc::clone(&join_schema),
    );
    let filtered = builder.add_filter(
        join,
        Expr::binary(Expr::column("score"), BinaryOp::Gt, Expr::column("avg_score")),
        join_schema,
    );
    let count_input = builder.add_projection(
        filtered,
        projection_outputs(vec![Expr::literal(LiteralValue::Int64(1)).alias("one")], one_schema),
    );
    let count = builder.add_aggregate(
        count_input,
        AggregateSpec::new(
            Vec::new(),
            vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("one"), "count")],
        ),
        Arc::clone(&count_schema),
    );
    let projection = builder.add_projection(count, projection_outputs(vec![Expr::column("count")], count_schema));
    builder.finish_with_host_sink(projection)
}

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
            builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::placeholder_row()))
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

/// Three group-only dedup aggregates combined by two stacked keyed LeftAnti
/// joins under an outermost groupless `count(*)`, i.e. `count(*) from ((a
/// except b) except c)`. Models a CHAIN of EXCEPT DISTINCTs rather than a
/// single one: the second anti join's probe side is the first anti join
/// itself, proven dedup transitively instead of being a dedup aggregate
/// directly.
pub(super) fn projection_root_setop_chained_dedup_scalar_plan() -> QueryPlan {
    let key_schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false)]));
    let one_schema = Arc::new(Schema::new(vec![Field::new("one", DataType::Int64, false)]));
    let scalar_schema = Arc::new(Schema::new(vec![Field::new("count", DataType::Int64, false)]));

    let mut builder = QueryPlanBuilder::new();
    let a = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/projection-setop-chained-a.parquet",
        Arc::clone(&key_schema),
        vec!["k".into()],
    )));
    let b = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/projection-setop-chained-b.parquet",
        Arc::clone(&key_schema),
        vec!["k".into()],
    )));
    let c = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/projection-setop-chained-c.parquet",
        Arc::clone(&key_schema),
        vec!["k".into()],
    )));
    let dedup_a =
        builder.add_aggregate(a, AggregateSpec::new(vec![Expr::column("k")], Vec::new()), Arc::clone(&key_schema));
    let dedup_b =
        builder.add_aggregate(b, AggregateSpec::new(vec![Expr::column("k")], Vec::new()), Arc::clone(&key_schema));
    let dedup_c =
        builder.add_aggregate(c, AggregateSpec::new(vec![Expr::column("k")], Vec::new()), Arc::clone(&key_schema));

    // EXCEPT DISTINCT is NULL-equality: NULL matches NULL. The single-anti-join
    // fixtures above use the `JoinSpec` default (`NullsDoNotMatch`); this chain
    // is the first fixture in this file to exercise `NullsMatch` on anti-join keys.
    let anti_ab = builder.add_join(
        dedup_a,
        dedup_b,
        JoinSpec::new(
            JoinType::LeftAnti,
            vec![JoinKey::new(ColumnRef::input(0, "k", "k"), ColumnRef::input(1, "k", "k"))],
        )
        .with_null_equality(JoinNullEquality::NullsMatch)
        .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "k", "k"), "k")])
        .with_build_side(JoinSide::Right),
        Arc::clone(&key_schema),
    );
    let anti_abc = builder.add_join(
        anti_ab,
        dedup_c,
        JoinSpec::new(
            JoinType::LeftAnti,
            vec![JoinKey::new(ColumnRef::input(0, "k", "k"), ColumnRef::input(1, "k", "k"))],
        )
        .with_null_equality(JoinNullEquality::NullsMatch)
        .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "k", "k"), "k")])
        .with_build_side(JoinSide::Right),
        Arc::clone(&key_schema),
    );

    let count_input = builder.add_projection(
        anti_abc,
        projection_outputs(vec![Expr::literal(LiteralValue::Int64(1)).alias("one")], one_schema),
    );
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

#[test]
fn composed_pipeline_keeps_scalar_fanout_cardinality_as_node_facts() {
    // The retired scalar-fanout cardinality proof survives as node facts, not
    // a plan-family reason: every bounded driver (literal row, scan limit,
    // max rows, fetch ≤ 1) records at most one output row on its own node,
    // and the scalar aggregates record theirs.
    for (label, plan) in [
        ("one-row-driver", projection_root_scalar_fanout_one_row_driver_plan()),
        ("scan-limit-driver", projection_root_scalar_fanout_scan_limit_driver_plan()),
        ("max-rows-driver", projection_root_scalar_fanout_max_rows_driver_plan()),
        ("limit-driver", projection_root_scalar_fanout_limit_driver_plan()),
    ] {
        let analysis = plan.capability_analysis(rapids_analysis_context());
        assert!(
            analysis.nodes.iter().any(|node| node.max_output_rows == Some(1)),
            "{label} should keep a one-row node fact"
        );
        assert!(
            bounded_pipeline_admission(&plan).unwrap().is_some(),
            "{label} scalar fanout stays executable through the composed pipeline"
        );
    }
}

#[test]
fn composed_pipeline_rejects_scalar_fanout_anti_join_at_node() {
    // The old plan-family cardinality proof rejected the anti-join fan-out
    // variant; now the join node capability owns the rejection while the
    // scalar aggregates keep their one-row facts.
    let plan = projection_root_scalar_fanout_unsupported_join_plan();
    let analysis = plan.capability_analysis(rapids_analysis_context());
    assert!(analysis.nodes.iter().any(|node| node.max_output_rows == Some(1)));
    let projection =
        bounded_pipeline_rejection_projection(&plan).unwrap().expect("anti-join fan-out should project a rejection");
    assert!(
        matches!(projection.owner, CapabilityRejectionOwner::Node(_)),
        "anti-join fan-out should reject on the node, got {:?}",
        projection.owner
    );
    assert_eq!(projection.reason, "unsupported_join_node_capability");
}

#[test]
fn composed_pipeline_rejects_non_dedup_left_anti_build_input() {
    // Mirror of the probe-side proof: an anti build against a distinct
    // aggregate claims EXCEPT DISTINCT over EXCEPT ALL multiplicity, so the
    // build edge requires a group-only dedup aggregate.
    let plan = projection_root_setop_distinct_aggregate_plan();
    let projection = bounded_pipeline_rejection_projection(&plan)
        .unwrap()
        .expect("distinct-aggregate anti build should project a rejection");
    assert!(
        matches!(projection.owner, CapabilityRejectionOwner::Edge { .. }),
        "distinct-aggregate build should reject on the edge, got {:?}",
        projection.owner
    );
    assert_eq!(projection.reason, "left_anti_build_requires_dedup_aggregate");
    assert!(projection.facts.contains_key("producer_aggregate_state"));
}

#[test]
fn composed_pipeline_admits_chained_except_distinct_dedup_scalar() {
    // The setop-dedup proof moved off the plan-family recognizer: aggregate
    // nodes carry `aggregate_state`, and each LeftAnti input edge checks it. A
    // CHAIN of EXCEPT DISTINCTs stays admitted because every build is a
    // group-only dedup aggregate and every probe is dedup-proven.
    let plan = projection_root_setop_chained_dedup_scalar_plan();
    let admission = bounded_pipeline_admission(&plan)
        .unwrap()
        .expect("chained EXCEPT DISTINCT dedup-scalar pipeline should be admitted");
    assert_eq!(admission.envelope(), CapabilityEnvelope::ComposedPipeline);
}

#[test]
fn composed_pipeline_rejects_non_dedup_left_anti_probe_input() {
    // The recursive probe-side proof must still respect its negative boundary:
    // a left-anti join whose probe side is a duplicate-preserving (EXCEPT
    // ALL-shaped) subtree, not a dedup aggregate or another dedup-proven anti
    // join, is rejected on the probe edge.
    let plan = projection_root_setop_dedup_scalar_non_dedup_left_anti_input_plan();
    let projection = bounded_pipeline_rejection_projection(&plan)
        .unwrap()
        .expect("non-dedup left-anti probe input should project a rejection");
    assert!(
        matches!(projection.owner, CapabilityRejectionOwner::Edge { .. }),
        "non-dedup probe input should reject on the edge, got {:?}",
        projection.owner
    );
    assert_eq!(projection.reason, "left_anti_probe_requires_dedup_proven_input");
    assert!(projection.facts.contains_key("producer_aggregate_state"));
}
