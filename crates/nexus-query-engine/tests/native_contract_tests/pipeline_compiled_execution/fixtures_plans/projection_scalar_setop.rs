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
