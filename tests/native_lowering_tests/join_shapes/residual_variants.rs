use super::*;

#[test]
fn hash_join_filter_literal_in_list_lowers_to_native_residual_predicate() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-hash-join-filter-in-list");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    write_join_left_parquet_file(&left_path);
    write_join_right_parquet_file(&right_path);
    let plan = hash_join_filter_in_list_aggregate_sort_limit_plan(parquet_scan(&left_path), parquet_scan(&right_path));

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.rewritten_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.rejected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let join = rewritten_native_join(&result);
    let residual = join.spec.residual_predicate.as_ref().expect("join filter should lower as residual predicate");
    assert!(matches!(
        residual,
        Expr::Binary { left, right, op: cudf_query_engine::BinaryOp::And }
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
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-hash-join-filter-cast");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    write_join_left_parquet_file(&left_path);
    write_join_right_parquet_file(&right_path);
    let plan = hash_join_filter_cast_aggregate_sort_limit_plan(parquet_scan(&left_path), parquet_scan(&right_path));

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.rewritten_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.rejected_count(), 0, "outcomes={:?}", result.report.outcomes());
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
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-hash-join-filter-decimal-cast");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    write_join_left_decimal_payload_parquet_file(&left_path);
    write_join_right_decimal_payload_parquet_file(&right_path);
    let plan = hash_join_filter_cast_aggregate_sort_limit_plan(parquet_scan(&left_path), parquet_scan(&right_path));

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.rewritten_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.rejected_count(), 0, "outcomes={:?}", result.report.outcomes());
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
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-hash-join-filter-decimal-materialized-cast");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    write_join_left_parquet_file(&left_path);
    write_join_right_parquet_file(&right_path);
    let plan = hash_join_filter_decimal_cast_materialization_aggregate_sort_limit_plan(
        parquet_scan(&left_path),
        parquet_scan(&right_path),
        JoinType::Inner,
    );

    let result = lower_physical_plan_to_native(plan);

    let join = rewritten_native_join(&result);
    let residual = join.spec.residual_predicate.as_ref().expect("join filter should lower as residual predicate");
    assert!(matches!(
        residual,
        Expr::Binary { left, right, .. }
            if matches!(
                left.as_ref(),
                Expr::Column(column) if column.to_string() == "input0.__cudf_join_residual_left_0"
            )
                && matches!(
                    right.as_ref(),
                    Expr::Column(column) if column.to_string() == "input1.__cudf_join_residual_right_0"
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
        "__cudf_join_residual_left_0",
        "l_payload",
    );
    assert_join_input_materializes_decimal_cast(
        &result.plans[0],
        join.inputs[1],
        "__cudf_join_residual_right_0",
        "r_payload",
    );
    let sources = native_local_parquet_sources(&result.plans[0]);
    assert_eq!(sources.len(), 2);
    assert_eq!(sources[0].read_columns, vec!["key".to_string(), "l_m00".to_string(), "l_payload".to_string()]);
    assert_eq!(sources[1].read_columns, vec!["key".to_string(), "r_m01".to_string(), "r_payload".to_string()]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn right_hash_join_filter_decimal_cast_materialization_uses_swapped_native_inputs() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-right-hash-join-filter-decimal-materialized-cast");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    write_join_left_parquet_file(&left_path);
    write_join_right_parquet_file(&right_path);
    let plan = hash_join_filter_decimal_cast_materialization_aggregate_sort_limit_plan(
        parquet_scan(&left_path),
        parquet_scan(&right_path),
        JoinType::Right,
    );

    let result = lower_physical_plan_to_native(plan);

    let join = rewritten_native_join(&result);
    assert_eq!(join.spec.join_type, NativeJoinType::Left);
    let residual = join.spec.residual_predicate.as_ref().expect("join filter should lower as residual predicate");
    assert!(matches!(
        residual,
        Expr::Binary { left, right, .. }
            if matches!(
                left.as_ref(),
                Expr::Column(column) if column.to_string() == "input1.__cudf_join_residual_left_0"
            )
                && matches!(
                    right.as_ref(),
                    Expr::Column(column) if column.to_string() == "input0.__cudf_join_residual_right_0"
                )
    ));
    assert_join_input_materializes_decimal_cast(
        &result.plans[0],
        join.inputs[0],
        "__cudf_join_residual_right_0",
        "r_payload",
    );
    assert_join_input_materializes_decimal_cast(
        &result.plans[0],
        join.inputs[1],
        "__cudf_join_residual_left_0",
        "l_payload",
    );

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn right_hash_join_lowers_to_swapped_native_left_join_spec() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-right-hash-join");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    write_join_left_parquet_file(&left_path);
    write_join_right_parquet_file(&right_path);
    let plan = hash_join_aggregate_sort_limit_plan_with_type(
        parquet_scan(&left_path),
        parquet_scan(&right_path),
        JoinType::Right,
    );

    let result = lower_physical_plan_to_native(plan);

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
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-right-semi-hash-join");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    write_join_left_parquet_file(&left_path);
    write_join_right_parquet_file(&right_path);
    let plan = right_only_hash_join_aggregate_sort_limit_plan(
        parquet_scan(&left_path),
        parquet_scan(&right_path),
        JoinType::RightSemi,
    );

    let result = lower_physical_plan_to_native(plan);

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
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-right-anti-hash-join");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    write_join_left_parquet_file(&left_path);
    write_join_right_parquet_file(&right_path);
    let plan = right_only_hash_join_aggregate_sort_limit_plan(
        parquet_scan(&left_path),
        parquet_scan(&right_path),
        JoinType::RightAnti,
    );

    let result = lower_physical_plan_to_native(plan);

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
fn phase5_nested_loop_join_filter_lowers_to_native_residual_predicate() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-nlj-filter");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    write_join_left_parquet_file(&left_path);
    write_join_right_parquet_file(&right_path);
    let plan = nested_loop_join_filter_aggregate_sort_limit_plan(
        parquet_scan(&left_path),
        parquet_scan(&right_path),
        JoinType::Inner,
    );

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.rewritten_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.rejected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let join = result.plans[0]
        .nodes()
        .iter()
        .find_map(|node| match &node.kind {
            PlanNode::Join(join) => Some(join),
            _ => None,
        })
        .expect("native plan should contain a join node");
    assert!(matches!(join.spec.join_type, cudf_query_engine::JoinType::Inner));
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
fn phase5_full_nested_loop_join_filter_lowers_to_native_conditional_full_join() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-full-nlj-filter");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    write_join_left_parquet_file(&left_path);
    write_join_right_parquet_file(&right_path);
    let plan = nested_loop_join_filter_aggregate_sort_limit_plan(
        parquet_scan(&left_path),
        parquet_scan(&right_path),
        JoinType::Full,
    );

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.rewritten_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.rejected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let join = result.plans[0]
        .nodes()
        .iter()
        .find_map(|node| match &node.kind {
            PlanNode::Join(join) => Some(join),
            _ => None,
        })
        .expect("native plan should contain a join node");
    assert!(matches!(join.spec.join_type, cudf_query_engine::JoinType::Full));
    assert!(join.spec.keys.is_empty());
    assert!(join.spec.residual_predicate.is_some());

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn phase5_inner_nested_loop_join_without_filter_lowers_to_native_cross_join() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-nlj-cross");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    write_join_left_parquet_file(&left_path);
    write_join_right_parquet_file(&right_path);
    let plan = nested_loop_join_without_filter_aggregate_sort_limit_plan(
        parquet_scan(&left_path),
        parquet_scan(&right_path),
        JoinType::Inner,
    );

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.rewritten_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.rejected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let join = result.plans[0]
        .nodes()
        .iter()
        .find_map(|node| match &node.kind {
            PlanNode::Join(join) => Some(join),
            _ => None,
        })
        .expect("native plan should contain a join node");
    assert!(matches!(join.spec.join_type, cudf_query_engine::JoinType::Cross));
    assert!(join.spec.keys.is_empty());
    assert!(join.spec.residual_predicate.is_none());

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn phase5_right_nested_loop_join_filter_lowers_to_swapped_native_left_join() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-right-nlj");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    write_join_left_parquet_file(&left_path);
    write_join_right_parquet_file(&right_path);
    let plan = nested_loop_join_filter_aggregate_sort_limit_plan(
        parquet_scan(&left_path),
        parquet_scan(&right_path),
        JoinType::Right,
    );

    let result = lower_physical_plan_to_native(plan);

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
