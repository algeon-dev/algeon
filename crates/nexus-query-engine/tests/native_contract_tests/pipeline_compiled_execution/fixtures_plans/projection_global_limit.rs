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

#[test]
fn projection_global_limit_admits_projection_over_join_without_aggregate() {
    let plan = projection_root_global_limit_join_without_aggregate_plan();
    assert_eq!(plan.nodes().iter().filter(|node| matches!(node.kind, PlanNode::Limit(_))).count(), 1);
    let analysis = plan.capability_analysis();
    assert!(analysis.admits_projection_global_limit_pipeline(), "{analysis:#?}");
    let admission = bounded_pipeline_admission(&plan)
        .unwrap()
        .expect("projection/global-limit join should be admitted");
    assert_eq!(admission.envelope(), nexus_query_engine::capability::CapabilityEnvelope::ProjectionGlobalLimit);
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
            let pushed_join_fetch =
                builder.add_limit(joined_input, LimitSpec::fetch(10), Arc::clone(&join_schema));
            (
                pushed_join_fetch,
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
