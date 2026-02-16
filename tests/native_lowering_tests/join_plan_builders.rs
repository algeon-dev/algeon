use super::*;

pub(super) fn hash_join_filter_aggregate_sort_limit_plan(
    left: Arc<dyn ExecutionPlan>,
    right: Arc<dyn ExecutionPlan>,
) -> Arc<dyn ExecutionPlan> {
    let left_schema = left.schema();
    let right_schema = right.schema();
    let filter_schema = Arc::new(Schema::new(vec![
        left_schema.field_with_name("l_payload").unwrap().clone(),
        right_schema.field_with_name("r_payload").unwrap().clone(),
    ]));
    let filter_expr = binary(
        col("l_payload", &filter_schema).unwrap(),
        Operator::Gt,
        col("r_payload", &filter_schema).unwrap(),
        &filter_schema,
    )
    .unwrap();
    let filter = JoinFilter::new(
        filter_expr,
        vec![
            ColumnIndex { index: left_schema.index_of("l_payload").unwrap(), side: JoinSide::Left },
            ColumnIndex { index: right_schema.index_of("r_payload").unwrap(), side: JoinSide::Right },
        ],
        filter_schema,
    );
    let join: Arc<dyn ExecutionPlan> = Arc::new(
        HashJoinExec::try_new(
            left,
            right,
            vec![(col("key", &left_schema).unwrap(), col("key", &right_schema).unwrap())],
            Some(filter),
            &JoinType::Inner,
            Some(vec![
                left_schema.index_of("key").unwrap(),
                left_schema.index_of("l_m00").unwrap(),
                left_schema.fields().len() + right_schema.index_of("r_m01").unwrap(),
            ]),
            PartitionMode::CollectLeft,
            NullEquality::NullEqualsNothing,
            false,
        )
        .unwrap(),
    );

    aggregate_sort_limit_join_plan(join)
}

pub(super) fn hash_join_filter_decimal_cast_materialization_aggregate_sort_limit_plan(
    left: Arc<dyn ExecutionPlan>,
    right: Arc<dyn ExecutionPlan>,
    join_type: JoinType,
) -> Arc<dyn ExecutionPlan> {
    let left_schema = left.schema();
    let right_schema = right.schema();
    let filter_schema = Arc::new(Schema::new(vec![
        left_schema.field_with_name("l_payload").unwrap().clone(),
        right_schema.field_with_name("r_payload").unwrap().clone(),
    ]));
    let left_cast: Arc<dyn PhysicalExpr> =
        Arc::new(CastExpr::new(col("l_payload", &filter_schema).unwrap(), DataType::Decimal128(10, 0), None));
    let right_cast: Arc<dyn PhysicalExpr> =
        Arc::new(CastExpr::new(col("r_payload", &filter_schema).unwrap(), DataType::Decimal128(10, 0), None));
    let filter_expr = binary(left_cast, Operator::Gt, right_cast, &filter_schema).unwrap();
    let filter = JoinFilter::new(
        filter_expr,
        vec![
            ColumnIndex { index: left_schema.index_of("l_payload").unwrap(), side: JoinSide::Left },
            ColumnIndex { index: right_schema.index_of("r_payload").unwrap(), side: JoinSide::Right },
        ],
        filter_schema,
    );
    let join: Arc<dyn ExecutionPlan> = Arc::new(
        HashJoinExec::try_new(
            left,
            right,
            vec![(col("key", &left_schema).unwrap(), col("key", &right_schema).unwrap())],
            Some(filter),
            &join_type,
            Some(vec![
                left_schema.index_of("key").unwrap(),
                left_schema.index_of("l_m00").unwrap(),
                left_schema.fields().len() + right_schema.index_of("r_m01").unwrap(),
            ]),
            PartitionMode::CollectLeft,
            NullEquality::NullEqualsNothing,
            false,
        )
        .unwrap(),
    );

    aggregate_sort_limit_join_plan(join)
}

pub(super) fn hash_join_filter_case_decimal_divide_aggregate_sort_limit_plan(
    left: Arc<dyn ExecutionPlan>,
    right: Arc<dyn ExecutionPlan>,
) -> Arc<dyn ExecutionPlan> {
    let left_schema = left.schema();
    let right_schema = right.schema();
    let filter_schema = Arc::new(Schema::new(vec![
        left_schema.field_with_name("l_payload").unwrap().clone(),
        right_schema.field_with_name("r_payload").unwrap().clone(),
    ]));
    let left_payload = col("l_payload", &filter_schema).unwrap();
    let right_payload = col("r_payload", &filter_schema).unwrap();
    let zero = lit(ScalarValue::Decimal128(Some(0), 15, 2));
    let left_case: Arc<dyn PhysicalExpr> = Arc::new(
        CaseExpr::try_new(
            None,
            vec![(
                binary(Arc::clone(&left_payload), Operator::Gt, Arc::clone(&zero), &filter_schema).unwrap(),
                binary(Arc::clone(&left_payload), Operator::Divide, Arc::clone(&left_payload), &filter_schema).unwrap(),
            )],
            None,
        )
        .unwrap(),
    );
    let right_case: Arc<dyn PhysicalExpr> = Arc::new(
        CaseExpr::try_new(
            None,
            vec![(
                binary(Arc::clone(&right_payload), Operator::Gt, Arc::clone(&zero), &filter_schema).unwrap(),
                binary(Arc::clone(&right_payload), Operator::Divide, Arc::clone(&right_payload), &filter_schema)
                    .unwrap(),
            )],
            None,
        )
        .unwrap(),
    );
    let filter_expr = binary(left_case, Operator::Gt, right_case, &filter_schema).unwrap();
    let filter = JoinFilter::new(
        filter_expr,
        vec![
            ColumnIndex { index: left_schema.index_of("l_payload").unwrap(), side: JoinSide::Left },
            ColumnIndex { index: right_schema.index_of("r_payload").unwrap(), side: JoinSide::Right },
        ],
        filter_schema,
    );
    let join: Arc<dyn ExecutionPlan> = Arc::new(
        HashJoinExec::try_new(
            left,
            right,
            vec![(col("key", &left_schema).unwrap(), col("key", &right_schema).unwrap())],
            Some(filter),
            &JoinType::Inner,
            Some(vec![
                left_schema.index_of("key").unwrap(),
                left_schema.index_of("l_m00").unwrap(),
                left_schema.fields().len() + right_schema.index_of("r_m01").unwrap(),
            ]),
            PartitionMode::CollectLeft,
            NullEquality::NullEqualsNothing,
            false,
        )
        .unwrap(),
    );

    aggregate_sort_limit_join_plan(join)
}

pub(super) fn hash_join_filter_substring_materialization_aggregate_sort_limit_plan(
    left: Arc<dyn ExecutionPlan>,
    right: Arc<dyn ExecutionPlan>,
) -> Arc<dyn ExecutionPlan> {
    let left_schema = left.schema();
    let right_schema = right.schema();
    let filter_schema = Arc::new(Schema::new(vec![
        left_schema.field_with_name("l_zip").unwrap().clone(),
        right_schema.field_with_name("r_zip").unwrap().clone(),
    ]));
    let left_substr: Arc<dyn PhysicalExpr> = Arc::new(ScalarFunctionExpr::new(
        "substr",
        unicode::substr(),
        vec![col("l_zip", &filter_schema).unwrap(), lit(ScalarValue::Int64(Some(1))), lit(ScalarValue::Int64(Some(5)))],
        Arc::new(Field::new("substr", DataType::Utf8View, false)),
        Arc::new(ConfigOptions::new()),
    ));
    let right_substr: Arc<dyn PhysicalExpr> = Arc::new(ScalarFunctionExpr::new(
        "substr",
        unicode::substr(),
        vec![col("r_zip", &filter_schema).unwrap(), lit(ScalarValue::Int64(Some(1))), lit(ScalarValue::Int64(Some(5)))],
        Arc::new(Field::new("substr", DataType::Utf8View, false)),
        Arc::new(ConfigOptions::new()),
    ));
    let zip_prefix_in_list = in_list(
        Arc::clone(&left_substr),
        vec![lit(ScalarValue::Utf8View(Some("12345".to_string())))],
        &false,
        filter_schema.as_ref(),
    )
    .unwrap();
    let zip_prefix_diff = binary(left_substr, Operator::NotEq, right_substr, &filter_schema).unwrap();
    let filter_expr = binary(zip_prefix_in_list, Operator::Or, zip_prefix_diff, &filter_schema).unwrap();
    let filter = JoinFilter::new(
        filter_expr,
        vec![
            ColumnIndex { index: left_schema.index_of("l_zip").unwrap(), side: JoinSide::Left },
            ColumnIndex { index: right_schema.index_of("r_zip").unwrap(), side: JoinSide::Right },
        ],
        filter_schema,
    );
    let join: Arc<dyn ExecutionPlan> = Arc::new(
        HashJoinExec::try_new(
            left,
            right,
            vec![(col("key", &left_schema).unwrap(), col("key", &right_schema).unwrap())],
            Some(filter),
            &JoinType::Inner,
            Some(vec![
                left_schema.index_of("key").unwrap(),
                left_schema.index_of("l_m00").unwrap(),
                left_schema.fields().len() + right_schema.index_of("r_m01").unwrap(),
            ]),
            PartitionMode::CollectLeft,
            NullEquality::NullEqualsNothing,
            false,
        )
        .unwrap(),
    );

    aggregate_sort_limit_join_plan(join)
}

pub(super) fn hash_join_filter_upper_cast_materialization_aggregate_sort_limit_plan(
    left: Arc<dyn ExecutionPlan>,
    right: Arc<dyn ExecutionPlan>,
) -> Arc<dyn ExecutionPlan> {
    let left_schema = left.schema();
    let right_schema = right.schema();
    let filter_schema = Arc::new(Schema::new(vec![
        left_schema.field_with_name("l_country").unwrap().clone(),
        right_schema.field_with_name("r_country").unwrap().clone(),
    ]));
    let upper_country: Arc<dyn PhysicalExpr> = Arc::new(ScalarFunctionExpr::new(
        "upper",
        string::upper(),
        vec![col("r_country", &filter_schema).unwrap()],
        Arc::new(Field::new("upper", DataType::Utf8, false)),
        Arc::new(ConfigOptions::new()),
    ));
    let cast_upper_country: Arc<dyn PhysicalExpr> = Arc::new(CastExpr::new(upper_country, DataType::Utf8View, None));
    let filter_expr =
        binary(col("l_country", &filter_schema).unwrap(), Operator::NotEq, cast_upper_country, &filter_schema).unwrap();
    let filter = JoinFilter::new(
        filter_expr,
        vec![
            ColumnIndex { index: left_schema.index_of("l_country").unwrap(), side: JoinSide::Left },
            ColumnIndex { index: right_schema.index_of("r_country").unwrap(), side: JoinSide::Right },
        ],
        filter_schema,
    );
    let join: Arc<dyn ExecutionPlan> = Arc::new(
        HashJoinExec::try_new(
            left,
            right,
            vec![(col("key", &left_schema).unwrap(), col("key", &right_schema).unwrap())],
            Some(filter),
            &JoinType::Inner,
            Some(vec![
                left_schema.index_of("key").unwrap(),
                left_schema.index_of("l_m00").unwrap(),
                left_schema.fields().len() + right_schema.index_of("r_m01").unwrap(),
            ]),
            PartitionMode::CollectLeft,
            NullEquality::NullEqualsNothing,
            false,
        )
        .unwrap(),
    );

    aggregate_sort_limit_join_plan(join)
}

pub(super) fn hash_join_filter_in_list_aggregate_sort_limit_plan(
    left: Arc<dyn ExecutionPlan>,
    right: Arc<dyn ExecutionPlan>,
) -> Arc<dyn ExecutionPlan> {
    let left_schema = left.schema();
    let right_schema = right.schema();
    let filter_schema = Arc::new(Schema::new(vec![
        left_schema.field_with_name("l_payload").unwrap().clone(),
        right_schema.field_with_name("r_payload").unwrap().clone(),
    ]));
    let in_list_expr = in_list(
        col("l_payload", &filter_schema).unwrap(),
        vec![lit(ScalarValue::Int32(Some(20))), lit(ScalarValue::Int32(Some(30)))],
        &false,
        filter_schema.as_ref(),
    )
    .unwrap();
    let comparison = binary(
        col("l_payload", &filter_schema).unwrap(),
        Operator::Gt,
        col("r_payload", &filter_schema).unwrap(),
        &filter_schema,
    )
    .unwrap();
    let filter_expr = binary(in_list_expr, Operator::And, comparison, &filter_schema).unwrap();
    let filter = JoinFilter::new(
        filter_expr,
        vec![
            ColumnIndex { index: left_schema.index_of("l_payload").unwrap(), side: JoinSide::Left },
            ColumnIndex { index: right_schema.index_of("r_payload").unwrap(), side: JoinSide::Right },
        ],
        filter_schema,
    );
    let join: Arc<dyn ExecutionPlan> = Arc::new(
        HashJoinExec::try_new(
            left,
            right,
            vec![(col("key", &left_schema).unwrap(), col("key", &right_schema).unwrap())],
            Some(filter),
            &JoinType::Inner,
            Some(vec![
                left_schema.index_of("key").unwrap(),
                left_schema.index_of("l_m00").unwrap(),
                left_schema.fields().len() + right_schema.index_of("r_m01").unwrap(),
            ]),
            PartitionMode::CollectLeft,
            NullEquality::NullEqualsNothing,
            false,
        )
        .unwrap(),
    );

    aggregate_sort_limit_join_plan(join)
}

pub(super) fn hash_join_filter_cast_aggregate_sort_limit_plan(
    left: Arc<dyn ExecutionPlan>,
    right: Arc<dyn ExecutionPlan>,
) -> Arc<dyn ExecutionPlan> {
    let left_schema = left.schema();
    let right_schema = right.schema();
    let filter_schema = Arc::new(Schema::new(vec![
        left_schema.field_with_name("l_payload").unwrap().clone(),
        right_schema.field_with_name("r_payload").unwrap().clone(),
    ]));
    let left_cast: Arc<dyn PhysicalExpr> =
        Arc::new(CastExpr::new(col("l_payload", &filter_schema).unwrap(), DataType::Float64, None));
    let right_cast: Arc<dyn PhysicalExpr> =
        Arc::new(CastExpr::new(col("r_payload", &filter_schema).unwrap(), DataType::Float64, None));
    let filter_expr = binary(left_cast, Operator::Gt, right_cast, &filter_schema).unwrap();
    let filter = JoinFilter::new(
        filter_expr,
        vec![
            ColumnIndex { index: left_schema.index_of("l_payload").unwrap(), side: JoinSide::Left },
            ColumnIndex { index: right_schema.index_of("r_payload").unwrap(), side: JoinSide::Right },
        ],
        filter_schema,
    );
    let join: Arc<dyn ExecutionPlan> = Arc::new(
        HashJoinExec::try_new(
            left,
            right,
            vec![(col("key", &left_schema).unwrap(), col("key", &right_schema).unwrap())],
            Some(filter),
            &JoinType::Inner,
            Some(vec![
                left_schema.index_of("key").unwrap(),
                left_schema.index_of("l_m00").unwrap(),
                left_schema.fields().len() + right_schema.index_of("r_m01").unwrap(),
            ]),
            PartitionMode::CollectLeft,
            NullEquality::NullEqualsNothing,
            false,
        )
        .unwrap(),
    );

    aggregate_sort_limit_join_plan(join)
}

pub(super) fn hash_join_aggregate_sort_limit_plan_with_type(
    left: Arc<dyn ExecutionPlan>,
    right: Arc<dyn ExecutionPlan>,
    join_type: JoinType,
) -> Arc<dyn ExecutionPlan> {
    let left_schema = left.schema();
    let right_schema = right.schema();
    let join: Arc<dyn ExecutionPlan> = Arc::new(
        HashJoinExec::try_new(
            left,
            right,
            vec![(col("key", &left_schema).unwrap(), col("key", &right_schema).unwrap())],
            None,
            &join_type,
            Some(vec![
                left_schema.index_of("key").unwrap(),
                left_schema.index_of("l_m00").unwrap(),
                left_schema.fields().len() + right_schema.index_of("r_m01").unwrap(),
            ]),
            PartitionMode::CollectLeft,
            NullEquality::NullEqualsNothing,
            false,
        )
        .unwrap(),
    );

    aggregate_sort_limit_join_plan(join)
}

pub(super) fn right_only_hash_join_aggregate_sort_limit_plan(
    left: Arc<dyn ExecutionPlan>,
    right: Arc<dyn ExecutionPlan>,
    join_type: JoinType,
) -> Arc<dyn ExecutionPlan> {
    let left_schema = left.schema();
    let right_schema = right.schema();
    let join: Arc<dyn ExecutionPlan> = Arc::new(
        HashJoinExec::try_new(
            left,
            right,
            vec![(col("key", &left_schema).unwrap(), col("key", &right_schema).unwrap())],
            None,
            &join_type,
            Some(vec![right_schema.index_of("key").unwrap(), right_schema.index_of("r_m01").unwrap()]),
            PartitionMode::CollectLeft,
            NullEquality::NullEqualsNothing,
            false,
        )
        .unwrap(),
    );

    aggregate_sort_limit_single_measure_join_plan(join, "r_m01")
}

pub(super) fn nested_loop_join_filter_aggregate_sort_limit_plan(
    left: Arc<dyn ExecutionPlan>,
    right: Arc<dyn ExecutionPlan>,
    join_type: JoinType,
) -> Arc<dyn ExecutionPlan> {
    let left_schema = left.schema();
    let right_schema = right.schema();
    let filter_schema = Arc::new(Schema::new(vec![
        left_schema.field_with_name("l_payload").unwrap().clone(),
        right_schema.field_with_name("r_payload").unwrap().clone(),
    ]));
    let filter_expr = binary(
        col("l_payload", &filter_schema).unwrap(),
        Operator::Gt,
        col("r_payload", &filter_schema).unwrap(),
        &filter_schema,
    )
    .unwrap();
    let filter = JoinFilter::new(
        filter_expr,
        vec![
            ColumnIndex { index: left_schema.index_of("l_payload").unwrap(), side: JoinSide::Left },
            ColumnIndex { index: right_schema.index_of("r_payload").unwrap(), side: JoinSide::Right },
        ],
        filter_schema,
    );
    let join: Arc<dyn ExecutionPlan> = Arc::new(
        NestedLoopJoinExec::try_new(
            left,
            right,
            Some(filter),
            &join_type,
            Some(vec![
                left_schema.index_of("key").unwrap(),
                left_schema.index_of("l_m00").unwrap(),
                left_schema.fields().len() + right_schema.index_of("r_m01").unwrap(),
            ]),
        )
        .unwrap(),
    );

    aggregate_sort_limit_join_plan(join)
}

pub(super) fn nested_loop_join_without_filter_aggregate_sort_limit_plan(
    left: Arc<dyn ExecutionPlan>,
    right: Arc<dyn ExecutionPlan>,
    join_type: JoinType,
) -> Arc<dyn ExecutionPlan> {
    let left_schema = left.schema();
    let right_schema = right.schema();
    let join: Arc<dyn ExecutionPlan> = Arc::new(
        NestedLoopJoinExec::try_new(
            left,
            right,
            None,
            &join_type,
            Some(vec![
                left_schema.index_of("key").unwrap(),
                left_schema.index_of("l_m00").unwrap(),
                left_schema.fields().len() + right_schema.index_of("r_m01").unwrap(),
            ]),
        )
        .unwrap(),
    );

    aggregate_sort_limit_join_plan(join)
}

pub(super) fn post_join_filter_aggregate_sort_limit_plan(
    left: Arc<dyn ExecutionPlan>,
    right: Arc<dyn ExecutionPlan>,
) -> Arc<dyn ExecutionPlan> {
    let left_schema = left.schema();
    let right_schema = right.schema();
    let join: Arc<dyn ExecutionPlan> = Arc::new(
        HashJoinExec::try_new(
            left,
            right,
            vec![(col("key", &left_schema).unwrap(), col("key", &right_schema).unwrap())],
            None,
            &JoinType::Inner,
            Some(vec![
                left_schema.index_of("key").unwrap(),
                left_schema.index_of("l_m00").unwrap(),
                left_schema.fields().len() + right_schema.index_of("r_m01").unwrap(),
            ]),
            PartitionMode::CollectLeft,
            NullEquality::NullEqualsNothing,
            false,
        )
        .unwrap(),
    );
    let join_schema = join.schema();
    let predicate =
        binary(col("l_m00", &join_schema).unwrap(), Operator::Gt, col("r_m01", &join_schema).unwrap(), &join_schema)
            .unwrap();
    let filter: Arc<dyn ExecutionPlan> = Arc::new(FilterExec::try_new(predicate, join).unwrap());

    aggregate_sort_limit_join_plan(filter)
}

pub(super) fn grouped_sum_join_input(
    input: Arc<dyn ExecutionPlan>,
    group_column: &str,
    measure_column: &str,
    aggregate_alias: &str,
) -> Arc<dyn ExecutionPlan> {
    let input_schema = input.schema();
    let group_by =
        PhysicalGroupBy::new_single(vec![(col(group_column, &input_schema).unwrap(), group_column.to_string())]);
    let sum = AggregateExprBuilder::new(sum_udaf(), vec![col(measure_column, &input_schema).unwrap()])
        .schema(Arc::clone(&input_schema))
        .alias(aggregate_alias)
        .build()
        .unwrap();
    Arc::new(
        AggregateExec::try_new(AggregateMode::Single, group_by, vec![Arc::new(sum)], vec![None], input, input_schema)
            .unwrap(),
    )
}

pub(super) fn aggregate_sort_limit_join_plan(input: Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
    let input_schema = input.schema();
    let x00 = binary(
        col("l_m00", &input_schema).unwrap(),
        Operator::Plus,
        col("r_m01", &input_schema).unwrap(),
        &input_schema,
    )
    .unwrap();
    let projection: Arc<dyn ExecutionPlan> = Arc::new(
        ProjectionExec::try_new(
            [
                ProjectionExpr { expr: col("key", &input_schema).unwrap(), alias: "key".to_string() },
                ProjectionExpr { expr: x00, alias: "x00".to_string() },
            ],
            input,
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
    let sort: Arc<dyn ExecutionPlan> = Arc::new(SortExec::new(sort_exprs, aggregate).with_fetch(Some(2)));
    Arc::new(LocalLimitExec::new(sort, 2))
}

pub(super) fn aggregate_sort_limit_single_measure_join_plan(
    input: Arc<dyn ExecutionPlan>,
    measure_column: &str,
) -> Arc<dyn ExecutionPlan> {
    let input_schema = input.schema();
    let projection: Arc<dyn ExecutionPlan> = Arc::new(
        ProjectionExec::try_new(
            [
                ProjectionExpr { expr: col("key", &input_schema).unwrap(), alias: "key".to_string() },
                ProjectionExpr { expr: col(measure_column, &input_schema).unwrap(), alias: "x00".to_string() },
            ],
            input,
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
    let sort: Arc<dyn ExecutionPlan> = Arc::new(SortExec::new(sort_exprs, aggregate).with_fetch(Some(2)));
    Arc::new(LocalLimitExec::new(sort, 2))
}
