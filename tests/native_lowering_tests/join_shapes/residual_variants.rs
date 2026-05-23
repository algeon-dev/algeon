use super::*;

fn timestamp_timezone_residual_hash_join_plan(
    left: Arc<dyn ExecutionPlan>,
    right: Arc<dyn ExecutionPlan>,
    include_in_list: bool,
) -> Arc<dyn ExecutionPlan> {
    let left_schema = left.schema();
    let right_schema = right.schema();
    let (filter_expr, filter_schema) =
        timestamp_timezone_residual_filter_expr(&left_schema, &right_schema, include_in_list);
    let filter =
        timestamp_timezone_residual_join_filter(filter_expr, Arc::clone(&filter_schema), &left_schema, &right_schema);
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

fn list_null_residual_hash_join_plan(
    left: Arc<dyn ExecutionPlan>,
    right: Arc<dyn ExecutionPlan>,
) -> Arc<dyn ExecutionPlan> {
    let left_schema = left.schema();
    let right_schema = right.schema();
    let filter_schema = Arc::new(Schema::new(vec![
        left_schema.field_with_name("l_tags").unwrap().clone(),
        right_schema.field_with_name("r_tags").unwrap().clone(),
    ]));
    let filter_expr = binary(
        is_null(col("l_tags", &filter_schema).unwrap()).unwrap(),
        Operator::Or,
        is_null(col("r_tags", &filter_schema).unwrap()).unwrap(),
        &filter_schema,
    )
    .unwrap();
    let filter = JoinFilter::new(
        filter_expr,
        vec![
            ColumnIndex { index: left_schema.index_of("l_tags").unwrap(), side: JoinSide::Left },
            ColumnIndex { index: right_schema.index_of("r_tags").unwrap(), side: JoinSide::Right },
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

fn integer_divide_residual_hash_join_plan(
    left: Arc<dyn ExecutionPlan>,
    right: Arc<dyn ExecutionPlan>,
) -> Arc<dyn ExecutionPlan> {
    let left_schema = left.schema();
    let right_schema = right.schema();
    let filter_schema = Arc::new(Schema::new(vec![
        left_schema.field_with_name("l_m00").unwrap().clone(),
        right_schema.field_with_name("r_m01").unwrap().clone(),
    ]));
    // Residual join predicates must reject integer divide before it reaches libcudf.
    let ratio = binary(
        col("l_m00", &filter_schema).unwrap(),
        Operator::Divide,
        col("r_m01", &filter_schema).unwrap(),
        &filter_schema,
    )
    .unwrap();
    let filter_expr = binary(ratio, Operator::Gt, lit(ScalarValue::Int64(Some(0))), &filter_schema).unwrap();
    let filter = JoinFilter::new(
        filter_expr,
        vec![
            ColumnIndex { index: left_schema.index_of("l_m00").unwrap(), side: JoinSide::Left },
            ColumnIndex { index: right_schema.index_of("r_m01").unwrap(), side: JoinSide::Right },
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

fn timestamp_timezone_residual_nested_loop_join_plan(
    left: Arc<dyn ExecutionPlan>,
    right: Arc<dyn ExecutionPlan>,
) -> Arc<dyn ExecutionPlan> {
    let left_schema = left.schema();
    let right_schema = right.schema();
    let (filter_expr, filter_schema) = timestamp_timezone_residual_filter_expr(&left_schema, &right_schema, false);
    let filter =
        timestamp_timezone_residual_join_filter(filter_expr, Arc::clone(&filter_schema), &left_schema, &right_schema);
    let join: Arc<dyn ExecutionPlan> = Arc::new(
        NestedLoopJoinExec::try_new(
            left,
            right,
            Some(filter),
            &JoinType::Inner,
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

fn timestamp_timezone_residual_filter_expr(
    left_schema: &SchemaRef,
    right_schema: &SchemaRef,
    include_in_list: bool,
) -> (Arc<dyn PhysicalExpr>, SchemaRef) {
    let filter_schema = Arc::new(Schema::new(vec![
        left_schema.field_with_name("l_event_ts").unwrap().clone(),
        right_schema.field_with_name("r_event_ts").unwrap().clone(),
    ]));
    let comparison = binary(
        col("l_event_ts", &filter_schema).unwrap(),
        Operator::Lt,
        col("r_event_ts", &filter_schema).unwrap(),
        &filter_schema,
    )
    .unwrap();
    if !include_in_list {
        return (comparison, filter_schema);
    }
    let in_list_expr = in_list(
        col("l_event_ts", &filter_schema).unwrap(),
        vec![lit(ScalarValue::TimestampMicrosecond(Some(100), Some("UTC".into())))],
        &false,
        filter_schema.as_ref(),
    )
    .unwrap();
    (binary(in_list_expr, Operator::And, comparison, &filter_schema).unwrap(), filter_schema)
}

fn timestamp_timezone_residual_join_filter(
    filter_expr: Arc<dyn PhysicalExpr>,
    filter_schema: SchemaRef,
    left_schema: &SchemaRef,
    right_schema: &SchemaRef,
) -> JoinFilter {
    JoinFilter::new(
        filter_expr,
        vec![
            ColumnIndex { index: left_schema.index_of("l_event_ts").unwrap(), side: JoinSide::Left },
            ColumnIndex { index: right_schema.index_of("r_event_ts").unwrap(), side: JoinSide::Right },
        ],
        filter_schema,
    )
}

#[test]
fn hash_join_filter_rejects_timestamp_timezone_residual_comparison() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-hash-join-filter-timestamp-tz");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    write_join_left_timestamp_tz_parquet_file(&left_path);
    write_join_right_timestamp_tz_parquet_file(&right_path);
    let plan = timestamp_timezone_residual_hash_join_plan(parquet_scan(&left_path), parquet_scan(&right_path), false);

    let result = lower_physical_plan_to_native_for_test(plan);

    assert_native_rejection(&result, CapabilityReason::UnsupportedType);
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn hash_join_filter_rejects_timestamp_timezone_residual_in_list() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-hash-join-filter-timestamp-tz-in-list");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    write_join_left_timestamp_tz_parquet_file(&left_path);
    write_join_right_timestamp_tz_parquet_file(&right_path);
    let plan = timestamp_timezone_residual_hash_join_plan(parquet_scan(&left_path), parquet_scan(&right_path), true);

    let result = lower_physical_plan_to_native_for_test(plan);

    assert_native_rejection(&result, CapabilityReason::UnsupportedType);
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn hash_join_filter_rejects_list_null_predicate_residual() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-hash-join-filter-list-null");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    write_join_left_list_parquet_file(&left_path);
    write_join_right_list_parquet_file(&right_path);
    let plan = list_null_residual_hash_join_plan(parquet_scan(&left_path), parquet_scan(&right_path));

    let result = lower_physical_plan_to_native_for_test(plan);

    assert_native_rejection(&result, CapabilityReason::UnsupportedType);
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn hash_join_filter_rejects_integer_divide_residual() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-hash-join-filter-integer-divide");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    write_join_left_parquet_file(&left_path);
    write_join_right_parquet_file(&right_path);
    let plan = integer_divide_residual_hash_join_plan(parquet_scan(&left_path), parquet_scan(&right_path));

    let result = lower_physical_plan_to_native_for_test(plan);

    assert_native_rejection(&result, CapabilityReason::UnsupportedExpression);
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn nested_loop_join_filter_rejects_timestamp_timezone_residual_comparison() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-nested-loop-filter-timestamp-tz");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    write_join_left_timestamp_tz_parquet_file(&left_path);
    write_join_right_timestamp_tz_parquet_file(&right_path);
    let plan = timestamp_timezone_residual_nested_loop_join_plan(parquet_scan(&left_path), parquet_scan(&right_path));

    let result = lower_physical_plan_to_native_for_test(plan);

    assert_native_rejection(&result, CapabilityReason::UnsupportedType);
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn hash_join_filter_literal_in_list_lowers_to_native_residual_predicate() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-hash-join-filter-in-list");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    write_join_left_parquet_file(&left_path);
    write_join_right_parquet_file(&right_path);
    let plan = hash_join_filter_in_list_aggregate_sort_limit_plan(parquet_scan(&left_path), parquet_scan(&right_path));

    let result = lower_physical_plan_to_native_for_test(plan);

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let join = rewritten_native_join(&result);
    let residual = join.spec.residual_predicate.as_ref().expect("join filter should lower as residual predicate");
    assert!(matches!(
        residual,
        Expr::Binary { left, right, op: nexus_query_engine::expr::BinaryOp::And, .. }
            if matches!(
                left.as_ref(),
                Expr::InList { expr, list, negated: false }
                    if matches!(expr.as_ref(), Expr::Column(column) if column.to_string() == "input0.l_payload")
                        && list.len() == 2
            )
                && matches!(
                    right.as_ref(),
                    Expr::Binary { left, right, .. }
                        if matches!(left.as_ref(), Expr::Column(column) if column.to_string() == "input0.l_payload")
                            && matches!(right.as_ref(), Expr::Column(column) if column.to_string() == "input1.r_payload")
                )
    ));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn hash_join_filter_cast_lowers_to_native_residual_predicate() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-hash-join-filter-cast");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    write_join_left_parquet_file(&left_path);
    write_join_right_parquet_file(&right_path);
    let plan = hash_join_filter_cast_aggregate_sort_limit_plan(parquet_scan(&left_path), parquet_scan(&right_path));

    let result = lower_physical_plan_to_native_for_test(plan);

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let join = rewritten_native_join(&result);
    let residual = join.spec.residual_predicate.as_ref().expect("join filter should lower as residual predicate");
    assert!(matches!(
        residual,
        Expr::Binary { left, right, .. }
            if matches!(
                left.as_ref(),
                Expr::Cast { expr, target_type }
                    if matches!(expr.as_ref(), Expr::Column(column) if column.to_string() == "input0.l_payload")
                        && matches!(target_type, DataType::Float64)
            )
                && matches!(
                    right.as_ref(),
                    Expr::Cast { expr, target_type }
                        if matches!(expr.as_ref(), Expr::Column(column) if column.to_string() == "input1.r_payload")
                            && matches!(target_type, DataType::Float64)
                )
    ));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn hash_join_filter_decimal_to_float_cast_lowers_to_native_residual_predicate() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-hash-join-filter-decimal-cast");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    write_join_left_decimal_payload_parquet_file(&left_path);
    write_join_right_decimal_payload_parquet_file(&right_path);
    let plan = hash_join_filter_cast_aggregate_sort_limit_plan(parquet_scan(&left_path), parquet_scan(&right_path));

    let result = lower_physical_plan_to_native_for_test(plan);

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let join = rewritten_native_join(&result);
    let residual = join.spec.residual_predicate.as_ref().expect("join filter should lower as residual predicate");
    assert!(matches!(
        residual,
        Expr::Binary { left, right, .. }
            if matches!(
                left.as_ref(),
                Expr::Cast { expr, target_type }
                    if matches!(expr.as_ref(), Expr::Column(column) if column.to_string() == "input0.l_payload")
                        && matches!(target_type, DataType::Float64)
            )
                && matches!(
                    right.as_ref(),
                    Expr::Cast { expr, target_type }
                        if matches!(expr.as_ref(), Expr::Column(column) if column.to_string() == "input1.r_payload")
                            && matches!(target_type, DataType::Float64)
                )
    ));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn hash_join_filter_decimal_cast_materializes_inputs_before_residual_predicate() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-hash-join-filter-decimal-materialized-cast");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    write_join_left_parquet_file(&left_path);
    write_join_right_parquet_file(&right_path);
    let plan = hash_join_filter_decimal_cast_materialization_aggregate_sort_limit_plan(
        parquet_scan(&left_path),
        parquet_scan(&right_path),
        JoinType::Inner,
    );

    let result = lower_physical_plan_to_native_for_test(plan);

    let join = rewritten_native_join(&result);
    let residual = join.spec.residual_predicate.as_ref().expect("join filter should lower as residual predicate");
    assert!(matches!(
        residual,
        Expr::Binary { left, right, .. }
            if matches!(
                left.as_ref(),
                Expr::Column(column) if column.to_string() == "input0.__nexus_native_join_residual_left_0"
            )
                && matches!(
                    right.as_ref(),
                    Expr::Column(column) if column.to_string() == "input1.__nexus_native_join_residual_right_0"
                )
    ));
    assert_join_output_projection(
        &join.spec.output_projection,
        &[
            (NativeJoinSide::Left, "input0.key", "key"),
            (NativeJoinSide::Left, "input0.l_m00", "l_m00"),
            (NativeJoinSide::Right, "input1.r_m01", "r_m01"),
        ],
    );
    assert_join_input_materializes_decimal_cast(
        &result.plans[0],
        join.inputs[0],
        "__nexus_native_join_residual_left_0",
        "l_payload",
    );
    assert_join_input_materializes_decimal_cast(
        &result.plans[0],
        join.inputs[1],
        "__nexus_native_join_residual_right_0",
        "r_payload",
    );
    let sources = native_local_parquet_sources(&result.plans[0]);
    assert_eq!(sources.len(), 2);
    assert_eq!(sources[0].read_columns, vec!["key".to_string(), "l_m00".to_string(), "l_payload".to_string()]);
    assert_eq!(sources[1].read_columns, vec!["key".to_string(), "r_m01".to_string(), "r_payload".to_string()]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn hash_join_filter_case_decimal_divide_materializes_inputs_before_residual_predicate() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-hash-join-filter-case-decimal-divide");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    write_join_left_decimal_payload_parquet_file(&left_path);
    write_join_right_decimal_payload_parquet_file(&right_path);
    let plan = hash_join_filter_case_decimal_divide_aggregate_sort_limit_plan(
        parquet_scan(&left_path),
        parquet_scan(&right_path),
    );

    let result = lower_physical_plan_to_native_for_test(plan);

    let join = rewritten_native_join(&result);
    let residual = join.spec.residual_predicate.as_ref().expect("join filter should lower as residual predicate");
    assert!(matches!(
        residual,
        Expr::Binary { left, right, .. }
            if matches!(
                left.as_ref(),
                Expr::Column(column) if column.to_string() == "input0.__nexus_native_join_residual_left_0"
            )
                && matches!(
                    right.as_ref(),
                    Expr::Column(column) if column.to_string() == "input1.__nexus_native_join_residual_right_0"
                )
    ));
    assert_join_input_materializes_case_decimal_divide(
        &result.plans[0],
        join.inputs[0],
        "__nexus_native_join_residual_left_0",
        "l_payload",
    );
    assert_join_input_materializes_case_decimal_divide(
        &result.plans[0],
        join.inputs[1],
        "__nexus_native_join_residual_right_0",
        "r_payload",
    );

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn hash_join_filter_substring_materializes_inputs_before_residual_predicate() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-hash-join-filter-substring-materialized");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    write_join_left_string_payload_parquet_file(&left_path);
    write_join_right_string_payload_parquet_file(&right_path);
    let plan = hash_join_filter_substring_materialization_aggregate_sort_limit_plan(
        parquet_scan(&left_path),
        parquet_scan(&right_path),
    );

    let result = lower_physical_plan_to_native_for_test(plan);

    let join = rewritten_native_join(&result);
    let residual = join.spec.residual_predicate.as_ref().expect("join filter should lower as residual predicate");
    assert!(matches!(
        residual,
        Expr::Binary { left, right, op: nexus_query_engine::expr::BinaryOp::Or, .. }
            if matches!(
                left.as_ref(),
                Expr::InList { expr, list, negated: false }
                    if matches!(
                        expr.as_ref(),
                        Expr::Column(column) if column.to_string() == "input0.__nexus_native_join_residual_left_0"
                    )
                        && list.len() == 1
            )
                && matches!(
                    right.as_ref(),
                    Expr::Binary { left, right, op: nexus_query_engine::expr::BinaryOp::NotEq, .. }
                        if matches!(
                            left.as_ref(),
                            Expr::Column(column) if column.to_string() == "input0.__nexus_native_join_residual_left_1"
                        )
                            && matches!(
                                right.as_ref(),
                                Expr::Column(column) if column.to_string() == "input1.__nexus_native_join_residual_right_0"
                            )
                )
    ));
    assert_join_input_materializes_substring(
        &result.plans[0],
        join.inputs[0],
        "__nexus_native_join_residual_left_0",
        "l_zip",
    );
    assert_join_input_materializes_substring(
        &result.plans[0],
        join.inputs[0],
        "__nexus_native_join_residual_left_1",
        "l_zip",
    );
    assert_join_input_materializes_substring(
        &result.plans[0],
        join.inputs[1],
        "__nexus_native_join_residual_right_0",
        "r_zip",
    );

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn hash_join_filter_upper_string_cast_materializes_input_before_residual_predicate() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-hash-join-filter-upper-cast-materialized");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    write_join_left_string_payload_parquet_file(&left_path);
    write_join_right_string_payload_parquet_file(&right_path);
    let plan = hash_join_filter_upper_cast_materialization_aggregate_sort_limit_plan(
        parquet_scan(&left_path),
        parquet_scan(&right_path),
    );

    let result = lower_physical_plan_to_native_for_test(plan);

    let join = rewritten_native_join(&result);
    let residual = join.spec.residual_predicate.as_ref().expect("join filter should lower as residual predicate");
    assert!(matches!(
        residual,
        Expr::Binary { left, right, op: nexus_query_engine::expr::BinaryOp::NotEq, .. }
            if matches!(left.as_ref(), Expr::Column(column) if column.to_string() == "input0.l_country")
                && matches!(
                    right.as_ref(),
                    Expr::Column(column) if column.to_string() == "input1.__nexus_native_join_residual_right_0"
                )
    ));
    assert_join_input_materializes_upper_utf8view_cast(
        &result.plans[0],
        join.inputs[1],
        "__nexus_native_join_residual_right_0",
        "r_country",
    );

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn right_hash_join_filter_decimal_cast_materialization_uses_swapped_native_inputs() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-right-hash-join-filter-decimal-materialized-cast");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    write_join_left_parquet_file(&left_path);
    write_join_right_parquet_file(&right_path);
    let plan = hash_join_filter_decimal_cast_materialization_aggregate_sort_limit_plan(
        parquet_scan(&left_path),
        parquet_scan(&right_path),
        JoinType::Right,
    );

    let result = lower_physical_plan_to_native_for_test(plan);

    let join = rewritten_native_join(&result);
    assert_eq!(join.spec.join_type, NativeJoinType::Left);
    let residual = join.spec.residual_predicate.as_ref().expect("join filter should lower as residual predicate");
    assert!(matches!(
        residual,
        Expr::Binary { left, right, .. }
            if matches!(
                left.as_ref(),
                Expr::Column(column) if column.to_string() == "input1.__nexus_native_join_residual_left_0"
            )
                && matches!(
                    right.as_ref(),
                    Expr::Column(column) if column.to_string() == "input0.__nexus_native_join_residual_right_0"
                )
    ));
    assert_join_input_materializes_decimal_cast(
        &result.plans[0],
        join.inputs[0],
        "__nexus_native_join_residual_right_0",
        "r_payload",
    );
    assert_join_input_materializes_decimal_cast(
        &result.plans[0],
        join.inputs[1],
        "__nexus_native_join_residual_left_0",
        "l_payload",
    );

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn right_hash_join_lowers_to_swapped_native_left_join_spec() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-right-hash-join");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    write_join_left_parquet_file(&left_path);
    write_join_right_parquet_file(&right_path);
    let plan = hash_join_aggregate_sort_limit_plan_with_type(
        parquet_scan(&left_path),
        parquet_scan(&right_path),
        JoinType::Right,
    );

    let result = lower_physical_plan_to_native_for_test(plan);

    let join = rewritten_native_join(&result);
    assert_right_hash_join_contract(
        join,
        NativeJoinType::Left,
        NativeJoinSide::Right,
        &[
            (NativeJoinSide::Right, "input1.key", "key"),
            (NativeJoinSide::Right, "input1.l_m00", "l_m00"),
            (NativeJoinSide::Left, "input0.r_m01", "r_m01"),
        ],
    );

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn right_semi_hash_join_lowers_to_swapped_native_left_semi_join_spec() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-right-semi-hash-join");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    write_join_left_parquet_file(&left_path);
    write_join_right_parquet_file(&right_path);
    let plan = right_only_hash_join_aggregate_sort_limit_plan(
        parquet_scan(&left_path),
        parquet_scan(&right_path),
        JoinType::RightSemi,
    );

    let result = lower_physical_plan_to_native_for_test(plan);

    let join = rewritten_native_join(&result);
    assert_right_hash_join_contract(
        join,
        NativeJoinType::LeftSemi,
        NativeJoinSide::Right,
        &[(NativeJoinSide::Left, "input0.key", "key"), (NativeJoinSide::Left, "input0.r_m01", "r_m01")],
    );

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn right_anti_hash_join_lowers_to_swapped_native_left_anti_join_spec() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-right-anti-hash-join");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    write_join_left_parquet_file(&left_path);
    write_join_right_parquet_file(&right_path);
    let plan = right_only_hash_join_aggregate_sort_limit_plan(
        parquet_scan(&left_path),
        parquet_scan(&right_path),
        JoinType::RightAnti,
    );

    let result = lower_physical_plan_to_native_for_test(plan);

    let join = rewritten_native_join(&result);
    assert_right_hash_join_contract(
        join,
        NativeJoinType::LeftAnti,
        NativeJoinSide::Right,
        &[(NativeJoinSide::Left, "input0.key", "key"), (NativeJoinSide::Left, "input0.r_m01", "r_m01")],
    );

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn right_semi_count_star_uses_native_preserved_side_for_hidden_row_presence() {
    assert_right_semi_anti_count_star_hidden_row_presence(JoinType::RightSemi, NativeJoinType::LeftSemi);
}

#[test]
fn right_anti_count_star_uses_native_preserved_side_for_hidden_row_presence() {
    assert_right_semi_anti_count_star_hidden_row_presence(JoinType::RightAnti, NativeJoinType::LeftAnti);
}

fn assert_right_semi_anti_count_star_hidden_row_presence(
    datafusion_join_type: JoinType,
    expected_native_join_type: NativeJoinType,
) {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-right-semi-anti-count-star");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    write_join_left_parquet_file(&left_path);
    write_join_right_parquet_file(&right_path);
    let plan = hash_join_count_star_plan(parquet_scan(&left_path), parquet_scan(&right_path), datafusion_join_type);

    let result = lower_physical_plan_to_native_for_test(plan);

    let native_plan = &result.plans[0];
    let join = rewritten_native_join(&result);
    assert_eq!(join.spec.join_type, expected_native_join_type);
    assert_eq!(join.spec.output_projection.len(), 1);
    assert_join_output_projection(
        &join.spec.output_projection,
        &[(NativeJoinSide::Left, "input0.key", "__nexus_native_join_row_presence")],
    );
    assert!(
        !native_plan.capability_analysis().has_rejections(),
        "capability analysis should accept hidden preserved-side semi/anti output"
    );

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn nested_loop_join_filter_lowers_to_native_residual_predicate() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-nlj-filter");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    write_join_left_parquet_file(&left_path);
    write_join_right_parquet_file(&right_path);
    let plan = nested_loop_join_filter_aggregate_sort_limit_plan(
        parquet_scan(&left_path),
        parquet_scan(&right_path),
        JoinType::Inner,
    );

    let result = lower_physical_plan_to_native_for_test(plan);

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let join = result.plans[0]
        .nodes()
        .iter()
        .find_map(|node| match &node.kind {
            PlanNode::Join(join) => Some(join),
            _ => None,
        })
        .expect("native plan should contain a join node");
    assert!(matches!(join.spec.join_type, nexus_query_engine::plan::JoinType::Inner));
    assert!(join.spec.keys.is_empty());
    let residual = join.spec.residual_predicate.as_ref().expect("nested loop filter should lower as residual");
    assert!(matches!(
        residual,
        Expr::Binary { left, right, .. }
            if matches!(left.as_ref(), Expr::Column(column) if column.to_string() == "input0.l_payload")
                && matches!(right.as_ref(), Expr::Column(column) if column.to_string() == "input1.r_payload")
    ));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn full_nested_loop_join_filter_lowers_to_native_conditional_full_join() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-full-nlj-filter");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    write_join_left_parquet_file(&left_path);
    write_join_right_parquet_file(&right_path);
    let plan = nested_loop_join_filter_aggregate_sort_limit_plan(
        parquet_scan(&left_path),
        parquet_scan(&right_path),
        JoinType::Full,
    );

    let result = lower_physical_plan_to_native_for_test(plan);

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let join = result.plans[0]
        .nodes()
        .iter()
        .find_map(|node| match &node.kind {
            PlanNode::Join(join) => Some(join),
            _ => None,
        })
        .expect("native plan should contain a join node");
    assert!(matches!(join.spec.join_type, nexus_query_engine::plan::JoinType::Full));
    assert!(join.spec.keys.is_empty());
    assert!(join.spec.residual_predicate.is_some());

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn inner_nested_loop_join_without_filter_lowers_to_native_cross_join() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-nlj-cross");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    write_join_left_parquet_file(&left_path);
    write_join_right_parquet_file(&right_path);
    let plan = nested_loop_join_without_filter_aggregate_sort_limit_plan(
        parquet_scan(&left_path),
        parquet_scan(&right_path),
        JoinType::Inner,
    );

    let result = lower_physical_plan_to_native_for_test(plan);

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let join = result.plans[0]
        .nodes()
        .iter()
        .find_map(|node| match &node.kind {
            PlanNode::Join(join) => Some(join),
            _ => None,
        })
        .expect("native plan should contain a join node");
    assert!(matches!(join.spec.join_type, nexus_query_engine::plan::JoinType::Cross));
    assert!(join.spec.keys.is_empty());
    assert!(join.spec.residual_predicate.is_none());

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn predicate_free_full_nested_loop_join_lowers_to_native_keyless_outer_cross_join() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-full-nlj-outer-cross");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    write_join_left_parquet_file(&left_path);
    write_join_right_parquet_file(&right_path);
    let plan = nested_loop_join_without_filter_aggregate_sort_limit_plan(
        parquet_scan(&left_path),
        parquet_scan(&right_path),
        JoinType::Full,
    );

    let result = lower_physical_plan_to_native_for_test(plan);

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let join = rewritten_native_join(&result);
    assert_eq!(join.spec.join_type, NativeJoinType::Full);
    assert_eq!(join.spec.build_side, NativeJoinSide::Left);
    assert!(join.spec.keys.is_empty());
    assert!(join.spec.residual_predicate.is_none());

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn predicate_free_right_nested_loop_join_lowers_to_swapped_native_keyless_outer_cross_join() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-right-nlj-outer-cross");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    write_join_left_parquet_file(&left_path);
    write_join_right_parquet_file(&right_path);
    let plan = nested_loop_join_without_filter_aggregate_sort_limit_plan(
        parquet_scan(&left_path),
        parquet_scan(&right_path),
        JoinType::Right,
    );

    let result = lower_physical_plan_to_native_for_test(plan);

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let join = rewritten_native_join(&result);
    assert_eq!(join.spec.join_type, NativeJoinType::Left);
    assert_eq!(join.spec.build_side, NativeJoinSide::Right);
    assert!(join.spec.keys.is_empty());
    assert!(join.spec.residual_predicate.is_none());

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn right_nested_loop_join_filter_lowers_to_swapped_native_left_join() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-right-nlj");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    write_join_left_parquet_file(&left_path);
    write_join_right_parquet_file(&right_path);
    let plan = nested_loop_join_filter_aggregate_sort_limit_plan(
        parquet_scan(&left_path),
        parquet_scan(&right_path),
        JoinType::Right,
    );

    let result = lower_physical_plan_to_native_for_test(plan);

    let join = rewritten_native_join(&result);
    assert_eq!(join.spec.join_type, NativeJoinType::Left);
    assert_eq!(join.spec.build_side, NativeJoinSide::Right);
    assert!(join.spec.keys.is_empty());
    let residual = join.spec.residual_predicate.as_ref().expect("nested loop filter should lower as residual");
    assert!(matches!(
        residual,
        Expr::Binary { left, right, .. }
            if matches!(left.as_ref(), Expr::Column(column) if column.to_string() == "input1.l_payload")
                && matches!(right.as_ref(), Expr::Column(column) if column.to_string() == "input0.r_payload")
    ));

    fs::remove_dir_all(dir).unwrap();
}
