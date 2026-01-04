use super::*;

pub(super) fn aggregate_sort_limit_plan(
    scan: Arc<dyn ExecutionPlan>,
    projection_operator: Operator,
) -> Arc<dyn ExecutionPlan> {
    let sort = aggregate_sort_plan(scan, projection_operator);
    Arc::new(LocalLimitExec::new(sort, 2))
}

pub(super) fn count_only_sort_limit_plan(scan: Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
    let scan_schema = scan.schema();
    let count = AggregateExprBuilder::new(count_udaf(), vec![lit(ScalarValue::Int64(Some(1)))])
        .schema(Arc::clone(&scan_schema))
        .alias("count_rows")
        .build()
        .unwrap();
    let aggregate: Arc<dyn ExecutionPlan> = Arc::new(
        AggregateExec::try_new(
            AggregateMode::Single,
            PhysicalGroupBy::new_single(Vec::new()),
            vec![Arc::new(count)],
            vec![None],
            scan,
            scan_schema,
        )
        .unwrap(),
    );

    let aggregate_schema = aggregate.schema();
    let sort_exprs = LexOrdering::new([PhysicalSortExpr::new(
        col("count_rows", &aggregate_schema).unwrap(),
        ArrowSortOptions::new(true, false),
    )])
    .unwrap();
    let sort: Arc<dyn ExecutionPlan> = Arc::new(SortExec::new(sort_exprs, aggregate).with_fetch(Some(1)));
    Arc::new(LocalLimitExec::new(sort, 1))
}

pub(super) fn aggregate_sort_limit_plan_with_filter(
    scan: Arc<dyn ExecutionPlan>,
    predicate: Arc<dyn PhysicalExpr>,
    projection_operator: Operator,
) -> Arc<dyn ExecutionPlan> {
    let sort = aggregate_sort_plan_with_filter(scan, predicate, projection_operator);
    Arc::new(LocalLimitExec::new(sort, 2))
}

pub(super) fn aggregate_sort_limit_plan_without_filter(
    scan: Arc<dyn ExecutionPlan>,
    projection_operator: Operator,
) -> Arc<dyn ExecutionPlan> {
    let sort = aggregate_sort_plan_without_filter(scan, projection_operator);
    Arc::new(LocalLimitExec::new(sort, 2))
}

pub(super) fn grouped_late_column_multi_sum_sort_limit_plan(scan: Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
    let scan_schema = scan.schema();
    let group_by = PhysicalGroupBy::new_single(vec![(col("payload", &scan_schema).unwrap(), "payload".to_string())]);
    let sum_m00 = AggregateExprBuilder::new(sum_udaf(), vec![col("m00", &scan_schema).unwrap()])
        .schema(Arc::clone(&scan_schema))
        .alias("sum_m00")
        .build()
        .unwrap();
    let sum_m01 = AggregateExprBuilder::new(sum_udaf(), vec![col("m01", &scan_schema).unwrap()])
        .schema(Arc::clone(&scan_schema))
        .alias("sum_m01")
        .build()
        .unwrap();
    let aggregate: Arc<dyn ExecutionPlan> = Arc::new(
        AggregateExec::try_new(
            AggregateMode::Single,
            group_by,
            vec![Arc::new(sum_m00), Arc::new(sum_m01)],
            vec![None, None],
            scan,
            scan_schema,
        )
        .unwrap(),
    );

    let aggregate_schema = aggregate.schema();
    let sort_exprs = LexOrdering::new([PhysicalSortExpr::new(
        col("payload", &aggregate_schema).unwrap(),
        ArrowSortOptions::new(false, false),
    )])
    .unwrap();
    let sort: Arc<dyn ExecutionPlan> = Arc::new(SortExec::new(sort_exprs, aggregate).with_fetch(Some(3)));
    Arc::new(LocalLimitExec::new(sort, 3))
}

pub(super) fn grouped_count_projection_plan(
    scan: Arc<dyn ExecutionPlan>,
    projection_expr: Arc<dyn PhysicalExpr>,
    projection_alias: &str,
) -> Arc<dyn ExecutionPlan> {
    let projection: Arc<dyn ExecutionPlan> = Arc::new(
        ProjectionExec::try_new([ProjectionExpr { expr: projection_expr, alias: projection_alias.to_string() }], scan)
            .unwrap(),
    );
    let projection_schema = projection.schema();
    let group_by = PhysicalGroupBy::new_single(vec![(
        col(projection_alias, &projection_schema).unwrap(),
        projection_alias.to_string(),
    )]);
    let count = AggregateExprBuilder::new(count_udaf(), vec![lit(ScalarValue::Int64(Some(1)))])
        .schema(Arc::clone(&projection_schema))
        .alias("count_rows")
        .build()
        .unwrap();
    let aggregate: Arc<dyn ExecutionPlan> = Arc::new(
        AggregateExec::try_new(
            AggregateMode::Single,
            group_by,
            vec![Arc::new(count)],
            vec![None],
            projection,
            projection_schema,
        )
        .unwrap(),
    );
    let aggregate_schema = aggregate.schema();
    let sort_exprs = LexOrdering::new([PhysicalSortExpr::new(
        col(projection_alias, &aggregate_schema).unwrap(),
        ArrowSortOptions::new(false, false),
    )])
    .unwrap();
    Arc::new(LocalLimitExec::new(Arc::new(SortExec::new(sort_exprs, aggregate).with_fetch(Some(2))), 2))
}

pub(super) fn aggregate_topk_plan(
    scan: Arc<dyn ExecutionPlan>,
    projection_operator: Operator,
    fetch: usize,
) -> Arc<dyn ExecutionPlan> {
    let sort = aggregate_sort_plan(scan, projection_operator);
    let sort = sort.as_any().downcast_ref::<SortExec>().expect("aggregate_sort_plan returns SortExec");
    Arc::new(sort.with_fetch(Some(fetch)))
}

pub(super) fn aggregate_sort_plan(
    scan: Arc<dyn ExecutionPlan>,
    projection_operator: Operator,
) -> Arc<dyn ExecutionPlan> {
    let scan_schema = scan.schema();
    let predicate =
        binary(col("payload", &scan_schema).unwrap(), Operator::GtEq, lit(ScalarValue::Int32(Some(10))), &scan_schema)
            .unwrap();
    aggregate_sort_plan_with_filter(scan, predicate, projection_operator)
}

pub(super) fn aggregate_sort_plan_with_filter(
    scan: Arc<dyn ExecutionPlan>,
    predicate: Arc<dyn PhysicalExpr>,
    projection_operator: Operator,
) -> Arc<dyn ExecutionPlan> {
    let filter: Arc<dyn ExecutionPlan> = Arc::new(FilterExec::try_new(predicate, scan).unwrap());

    let filter_schema = filter.schema();
    let x00 = binary(
        col("m00", &filter_schema).unwrap(),
        projection_operator,
        col("m01", &filter_schema).unwrap(),
        &filter_schema,
    )
    .unwrap();
    let projection: Arc<dyn ExecutionPlan> = Arc::new(
        ProjectionExec::try_new(
            [
                ProjectionExpr { expr: col("key", &filter_schema).unwrap(), alias: "key".to_string() },
                ProjectionExpr { expr: x00, alias: "x00".to_string() },
            ],
            filter,
        )
        .unwrap(),
    );

    let projection_schema = projection.schema();
    let group_by = PhysicalGroupBy::new_single(vec![(col("key", &projection_schema).unwrap(), "key".to_string())]);
    let sum = AggregateExprBuilder::new(sum_udaf(), vec![col("x00", &projection_schema).unwrap()])
        .schema(Arc::clone(&projection_schema))
        .alias("sum_x00")
        .build()
        .unwrap();
    let aggregate: Arc<dyn ExecutionPlan> = Arc::new(
        AggregateExec::try_new(
            AggregateMode::Single,
            group_by,
            vec![Arc::new(sum)],
            vec![None],
            projection,
            projection_schema,
        )
        .unwrap(),
    );

    let aggregate_schema = aggregate.schema();
    let sort_exprs = LexOrdering::new([
        PhysicalSortExpr::new(col("sum_x00", &aggregate_schema).unwrap(), ArrowSortOptions::new(true, false)),
        PhysicalSortExpr::new(col("key", &aggregate_schema).unwrap(), ArrowSortOptions::new(false, false)),
    ])
    .unwrap();
    Arc::new(SortExec::new(sort_exprs, aggregate))
}

pub(super) fn aggregate_sort_limit_plan_with_projected_x00(
    scan: Arc<dyn ExecutionPlan>,
    predicate: Arc<dyn PhysicalExpr>,
    x00: Arc<dyn PhysicalExpr>,
) -> Arc<dyn ExecutionPlan> {
    let filter: Arc<dyn ExecutionPlan> = Arc::new(FilterExec::try_new(predicate, scan).unwrap());

    let filter_schema = filter.schema();
    let projection: Arc<dyn ExecutionPlan> = Arc::new(
        ProjectionExec::try_new(
            [
                ProjectionExpr { expr: col("key", &filter_schema).unwrap(), alias: "key".to_string() },
                ProjectionExpr { expr: x00, alias: "x00".to_string() },
            ],
            filter,
        )
        .unwrap(),
    );

    let projection_schema = projection.schema();
    let group_by = PhysicalGroupBy::new_single(vec![(col("key", &projection_schema).unwrap(), "key".to_string())]);
    let sum = AggregateExprBuilder::new(sum_udaf(), vec![col("x00", &projection_schema).unwrap()])
        .schema(Arc::clone(&projection_schema))
        .alias("sum_x00")
        .build()
        .unwrap();
    let aggregate: Arc<dyn ExecutionPlan> = Arc::new(
        AggregateExec::try_new(
            AggregateMode::Single,
            group_by,
            vec![Arc::new(sum)],
            vec![None],
            projection,
            projection_schema,
        )
        .unwrap(),
    );

    let aggregate_schema = aggregate.schema();
    let sort_exprs = LexOrdering::new([
        PhysicalSortExpr::new(col("sum_x00", &aggregate_schema).unwrap(), ArrowSortOptions::new(true, false)),
        PhysicalSortExpr::new(col("key", &aggregate_schema).unwrap(), ArrowSortOptions::new(false, false)),
    ])
    .unwrap();
    let sort: Arc<dyn ExecutionPlan> = Arc::new(SortExec::new(sort_exprs, aggregate));
    Arc::new(LocalLimitExec::new(sort, 2))
}

pub(super) fn aggregate_sort_plan_without_filter(
    scan: Arc<dyn ExecutionPlan>,
    projection_operator: Operator,
) -> Arc<dyn ExecutionPlan> {
    let scan_schema = scan.schema();
    let x00 =
        binary(col("m00", &scan_schema).unwrap(), projection_operator, col("m01", &scan_schema).unwrap(), &scan_schema)
            .unwrap();
    let projection: Arc<dyn ExecutionPlan> = Arc::new(
        ProjectionExec::try_new(
            [
                ProjectionExpr { expr: col("key", &scan_schema).unwrap(), alias: "key".to_string() },
                ProjectionExpr { expr: x00, alias: "x00".to_string() },
            ],
            scan,
        )
        .unwrap(),
    );

    let projection_schema = projection.schema();
    let group_by = PhysicalGroupBy::new_single(vec![(col("key", &projection_schema).unwrap(), "key".to_string())]);
    let sum = AggregateExprBuilder::new(sum_udaf(), vec![col("x00", &projection_schema).unwrap()])
        .schema(Arc::clone(&projection_schema))
        .alias("sum_x00")
        .build()
        .unwrap();
    let aggregate: Arc<dyn ExecutionPlan> = Arc::new(
        AggregateExec::try_new(
            AggregateMode::Single,
            group_by,
            vec![Arc::new(sum)],
            vec![None],
            projection,
            projection_schema,
        )
        .unwrap(),
    );

    let aggregate_schema = aggregate.schema();
    let sort_exprs = LexOrdering::new([
        PhysicalSortExpr::new(col("sum_x00", &aggregate_schema).unwrap(), ArrowSortOptions::new(true, false)),
        PhysicalSortExpr::new(col("key", &aggregate_schema).unwrap(), ArrowSortOptions::new(false, false)),
    ])
    .unwrap();
    Arc::new(SortExec::new(sort_exprs, aggregate))
}
