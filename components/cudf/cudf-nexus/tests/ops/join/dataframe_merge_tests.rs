//! Tests for the high-level `DataFrame::merge` API and `MergeOptions` / `JoinType`.
use cudf::{
    DataFrame, DataFrameCrossJoinOptions, DataFrameFilteredJoinOptions, DataFrameHashJoinOptions,
    DataFrameHashJoinPairIndices, DataFrameJoinIndices, DataFrameJoinOptions, DataFrameJoinPairIndices,
    DataFrameMixedJoinOptions, ErrorCode, ErrorKind, ExecutionOverrides, JoinType, MergeOptions, TypeId,
    frame::dataframe::InfoDetail,
    ops::{
        ast::AstExpression,
        join::{
            FilteredJoin, FilteredJoinOptions, HashJoinKeyMetadata, HashJoinStringOffsetWidth,
            filtered_join_allocation_preflight_for_metadata, filtered_join_probe_allocation_preflight,
        },
    },
    testing::prelude::*,
};

use crate::common::{
    assert_duplicate_name_error, assert_missing_column_error, column_from_slice, column_from_slice_nullable,
    column_from_strs_nullable, dependency_wait_call_count_for_stream, error_context_value, last_exec_mr_value,
    last_exec_stream_value, last_sync_stream_value, reset_dependency_observation, reset_exec_observation,
    reset_sync_observation, sample_runtime,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn left_df() -> DataFrame {
    // id: [1, 2, 3], score: [10, 20, 30]
    let ids = column_from_slice(&[1i32, 2, 3]).unwrap();
    let scores = column_from_slice(&[10i32, 20, 30]).unwrap();
    crate::common::dataframe_from_columns(vec!["id".to_string(), "score".to_string()], vec![ids, scores]).unwrap()
}

fn right_df() -> DataFrame {
    // id: [2, 3, 4], value: [200, 300, 400]
    let ids = column_from_slice(&[2i32, 3, 4]).unwrap();
    let values = column_from_slice(&[200i32, 300, 400]).unwrap();
    crate::common::dataframe_from_columns(vec!["id".to_string(), "value".to_string()], vec![ids, values]).unwrap()
}

fn left_df_asym() -> DataFrame {
    let ids = column_from_slice(&[1i32, 2, 3]).unwrap();
    let scores = column_from_slice(&[10i32, 20, 30]).unwrap();
    crate::common::dataframe_from_columns(vec!["left_id".to_string(), "score".to_string()], vec![ids, scores]).unwrap()
}

fn right_df_asym() -> DataFrame {
    let ids = column_from_slice(&[2i32, 3, 4]).unwrap();
    let values = column_from_slice(&[200i32, 300, 400]).unwrap();
    crate::common::dataframe_from_columns(vec!["right_id".to_string(), "value".to_string()], vec![ids, values]).unwrap()
}

fn nullable_i32_column(df: &DataFrame, name: &str) -> Vec<Option<i32>> {
    df.series(name).unwrap().to_optional_vec::<i32>().unwrap()
}

fn i32_column(df: &DataFrame, name: &str) -> Vec<i32> {
    df.series(name).unwrap().to_optional_vec::<i32>().unwrap().into_iter().map(|v| v.unwrap()).collect()
}

fn device_allocated_bytes(df: &DataFrame) -> usize {
    df.info_with(InfoDetail::DEVICE_ALLOCATED).unwrap().memory_accounting.unwrap().device_allocated_bytes.unwrap()
}

fn assert_merge_validation_error(err: &cudf::Error, operation: &str, reason: &str, remedy: &str) {
    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(err.operation(), operation);
    assert_eq!(error_context_value(err, "reason").as_deref(), Some(reason));
    assert_eq!(error_context_value(err, "remedy").as_deref(), Some(remedy));
}

fn cross_left_df(rt: &cudf::Runtime) -> DataFrame {
    DataFrame::from_series([
        ("id", rt.series(&[1i32, 2]).unwrap()),
        ("left_val", rt.series(&[10i32, 20]).unwrap()),
        ("unused_left", rt.series(&[1000i32, 2000]).unwrap()),
    ])
    .unwrap()
}

fn cross_right_df(rt: &cudf::Runtime) -> DataFrame {
    DataFrame::from_series([
        ("id", rt.series(&[100i32, 200, 300]).unwrap()),
        ("right_val", rt.series(&[7i32, 8, 9]).unwrap()),
        ("unused_right", rt.series(&[70i32, 80, 90]).unwrap()),
    ])
    .unwrap()
}

fn assert_single_side_indices_reservation_matches_allocated(indices: &DataFrameJoinIndices) {
    assert_eq!(
        DataFrameJoinIndices::reservation_size_bytes_for_len(indices.len()).unwrap(),
        indices.device_allocated_size_bytes().unwrap()
    );
}

fn left_less_than_right_predicate() -> AstExpression {
    AstExpression::less(AstExpression::left_column(0).unwrap(), AstExpression::right_column(0).unwrap()).unwrap()
}

fn sorted_pair_indices(indices: &DataFrameJoinPairIndices) -> Vec<(i32, i32)> {
    let left = indices.left().indices().to_optional_vec::<i32>().unwrap();
    let right = indices.right().indices().to_optional_vec::<i32>().unwrap();
    let mut pairs: Vec<_> = left.into_iter().zip(right).map(|(left, right)| (left.unwrap(), right.unwrap())).collect();
    pairs.sort();
    pairs
}

// ---------------------------------------------------------------------------
// Query-planning semi/anti helpers and late materialization
// ---------------------------------------------------------------------------

fn test_dataframe_left_semi_join_materializes_left_rows_without_right_duplicate_growth() {
    let rt = sample_runtime();
    let left = DataFrame::from_series([
        ("id", rt.series(&[1i32, 2, 2, 3, 4]).unwrap()),
        ("score", rt.series(&[10i32, 20, 21, 30, 40]).unwrap()),
    ])
    .unwrap();
    let right = DataFrame::from_series([
        ("id", rt.series(&[2i32, 2, 4]).unwrap()),
        ("right_payload", rt.series(&[200i32, 201, 400]).unwrap()),
    ])
    .unwrap();

    let indices = left.left_semi_join_indices(&right, DataFrameJoinOptions::new().on(["id"])).unwrap();
    let result = left.left_semi_join(&right, DataFrameJoinOptions::new().on(["id"])).unwrap();

    assert_single_side_indices_reservation_matches_allocated(&indices);
    assert_eq!(result.column_names(), vec!["id", "score"]);
    assert_eq!(result.num_rows(), 3);
    let mut rows: Vec<_> = i32_column(&result, "id").into_iter().zip(i32_column(&result, "score")).collect();
    rows.sort();
    assert_eq!(rows, vec![(2, 20), (2, 21), (4, 40)]);
}

fn test_dataframe_left_anti_join_materializes_only_non_matching_left_rows() {
    let rt = sample_runtime();
    let left = DataFrame::from_series([
        ("id", rt.series(&[1i32, 2, 2, 3, 4]).unwrap()),
        ("score", rt.series(&[10i32, 20, 21, 30, 40]).unwrap()),
    ])
    .unwrap();
    let right = DataFrame::from_series([("id", rt.series(&[2i32, 4]).unwrap())]).unwrap();

    let indices = left.left_anti_join_indices(&right, DataFrameJoinOptions::new().on(["id"])).unwrap();
    let result = left.left_anti_join(&right, DataFrameJoinOptions::new().on(["id"])).unwrap();

    assert_single_side_indices_reservation_matches_allocated(&indices);
    assert_eq!(result.column_names(), vec!["id", "score"]);
    assert_eq!(result.num_rows(), 2);
    let mut rows: Vec<_> = i32_column(&result, "id").into_iter().zip(i32_column(&result, "score")).collect();
    rows.sort();
    assert_eq!(rows, vec![(1, 10), (3, 30)]);
}

fn test_dataframe_join_indices_materialize_selected_payload_columns() {
    let rt = sample_runtime();
    let left = DataFrame::from_series([
        ("id", rt.series(&[1i32, 2, 3, 4]).unwrap()),
        ("score", rt.series(&[10i32, 20, 30, 40]).unwrap()),
        ("payload", rt.series(&[100i32, 200, 300, 400]).unwrap()),
    ])
    .unwrap();
    let right = DataFrame::from_series([("id", rt.series(&[2i32, 4]).unwrap())]).unwrap();

    let indices = left.left_semi_join_indices(&right, DataFrameJoinOptions::new().on(["id"])).unwrap();
    let materialized = left.materialize_join_indices(&indices, ["payload"]).unwrap();

    assert_eq!(indices.len(), 2);
    assert_eq!(materialized.column_names(), vec!["payload"]);
    assert_eq!(materialized.num_rows(), 2);
    let mut payload = i32_column(&materialized, "payload");
    payload.sort();
    assert_eq!(payload, vec![200, 400]);
}

fn test_dataframe_join_indices_materialization_reservation_matches_output_allocation() {
    let rt = sample_runtime();
    let left = DataFrame::from_series([
        ("id", rt.series(&[1i32, 2, 3, 4]).unwrap()),
        ("score", rt.series(&[10i32, 20, 30, 40]).unwrap()),
        ("payload", rt.series(&[100i32, 200, 300, 400]).unwrap()),
    ])
    .unwrap();
    let right = DataFrame::from_series([("id", rt.series(&[2i32, 4]).unwrap())]).unwrap();

    let indices = left.left_semi_join_indices(&right, DataFrameJoinOptions::new().on(["id"])).unwrap();
    let reservation = left.materialize_join_indices_reservation_size_bytes(&indices, ["payload"]).unwrap();
    let materialized = left.materialize_join_indices(&indices, ["payload"]).unwrap();

    assert_eq!(reservation, device_allocated_bytes(&materialized));
    assert!(reservation > 0);
}

fn test_dataframe_join_indices_materialization_reservation_is_zero_for_empty_gather_map() {
    let rt = sample_runtime();
    let left = DataFrame::from_series([
        ("id", rt.series(&[1i32, 2, 3, 4]).unwrap()),
        ("score", rt.series(&[10i32, 20, 30, 40]).unwrap()),
        ("payload", rt.series(&[100i32, 200, 300, 400]).unwrap()),
    ])
    .unwrap();
    let right = DataFrame::from_series([("id", rt.series(&[999i32]).unwrap())]).unwrap();

    let indices = left.left_semi_join_indices(&right, DataFrameJoinOptions::new().on(["id"])).unwrap();
    let reservation = left.materialize_join_indices_reservation_size_bytes(&indices, ["payload"]).unwrap();
    let materialized = left.materialize_join_indices(&indices, ["payload"]).unwrap();

    assert!(indices.is_empty());
    assert_eq!(reservation, 0);
    assert_eq!(device_allocated_bytes(&materialized), 0);
}

fn test_projected_cross_join_plan_reports_size_and_materializes_both_sides() {
    let rt = sample_runtime();
    let left = cross_left_df(&rt);
    let right = cross_right_df(&rt);

    let plan = left
        .projected_cross_join_plan(
            &right,
            DataFrameCrossJoinOptions::new()
                .left_projection([("id", "left_id"), ("left_val", "left_val")])
                .right_projection([("id", "right_id"), ("right_val", "right_val")]),
        )
        .unwrap();

    assert_eq!(plan.output_rows(), 6);
    assert_eq!(plan.output_column_names(), ["left_id", "left_val", "right_id", "right_val"]);
    let reservation = plan.reservation_size_bytes().unwrap();
    let materialized = plan.materialize().unwrap();

    assert_eq!(materialized.num_rows(), 6);
    assert_eq!(materialized.column_names(), vec!["left_id", "left_val", "right_id", "right_val"]);
    assert_eq!(i32_column(&materialized, "left_id"), vec![1, 1, 1, 2, 2, 2]);
    assert_eq!(i32_column(&materialized, "left_val"), vec![10, 10, 10, 20, 20, 20]);
    assert_eq!(i32_column(&materialized, "right_id"), vec![100, 200, 300, 100, 200, 300]);
    assert_eq!(i32_column(&materialized, "right_val"), vec![7, 8, 9, 7, 8, 9]);
    assert_eq!(reservation, device_allocated_bytes(&materialized));
    assert!(reservation > 0);
}

fn test_projected_cross_join_plan_supports_left_only_projection() {
    let rt = sample_runtime();
    let left = cross_left_df(&rt);
    let right = cross_right_df(&rt);

    let plan = left
        .projected_cross_join_plan(
            &right,
            DataFrameCrossJoinOptions::new().left_projection([("left_val", "left_only")]),
        )
        .unwrap();
    let reservation = plan.reservation_size_bytes().unwrap();
    let materialized = plan.materialize().unwrap();

    assert_eq!(plan.output_rows(), 6);
    assert_eq!(materialized.column_names(), vec!["left_only"]);
    assert_eq!(i32_column(&materialized, "left_only"), vec![10, 10, 10, 20, 20, 20]);
    assert_eq!(reservation, device_allocated_bytes(&materialized));
}

fn test_projected_cross_join_plan_supports_right_only_projection() {
    let rt = sample_runtime();
    let left = cross_left_df(&rt);
    let right = cross_right_df(&rt);

    let plan = left
        .projected_cross_join_plan(
            &right,
            DataFrameCrossJoinOptions::new().right_projection([("right_val", "right_only")]),
        )
        .unwrap();
    let reservation = plan.reservation_size_bytes().unwrap();
    let materialized = plan.materialize().unwrap();

    assert_eq!(plan.output_rows(), 6);
    assert_eq!(materialized.column_names(), vec!["right_only"]);
    assert_eq!(i32_column(&materialized, "right_only"), vec![7, 8, 9, 7, 8, 9]);
    assert_eq!(reservation, device_allocated_bytes(&materialized));
}

fn test_legacy_projected_cross_join_supports_one_sided_projection() {
    let rt = sample_runtime();
    let left = cross_left_df(&rt);
    let right = cross_right_df(&rt);

    let left_only = left
        .cross_join_projected(&right, DataFrameCrossJoinOptions::new().left_projection([("left_val", "left_only")]))
        .unwrap();
    let right_only = left
        .cross_join_projected(&right, DataFrameCrossJoinOptions::new().right_projection([("right_val", "right_only")]))
        .unwrap();

    assert_eq!(left_only.column_names(), vec!["left_only"]);
    assert_eq!(i32_column(&left_only, "left_only"), vec![10, 10, 10, 20, 20, 20]);
    assert_eq!(right_only.column_names(), vec!["right_only"]);
    assert_eq!(i32_column(&right_only, "right_only"), vec![7, 8, 9, 7, 8, 9]);
}

fn test_projected_cross_join_plan_reservation_and_materialize_use_execution_override() {
    let data_rt = sample_runtime();
    let exec_rt = sample_runtime();
    let left = cross_left_df(&data_rt);
    let right = cross_right_df(&data_rt);
    let ctx = ExecutionOverrides::new().with_stream(exec_rt.stream()).with_memory_resource(exec_rt.memory_resource());
    let plan = left
        .projected_cross_join_plan(
            &right,
            DataFrameCrossJoinOptions::new()
                .left_projection([("left_val", "left_val")])
                .right_projection([("right_val", "right_val")]),
        )
        .unwrap();

    reset_exec_observation();
    let reservation = plan.reservation_size_bytes_in(&ctx).unwrap();
    assert_eq!(last_exec_stream_value(), exec_rt.stream().testing_raw_handle_value());
    assert_eq!(last_exec_mr_value(), exec_rt.memory_resource().testing_raw_handle_value());

    reset_exec_observation();
    let materialized = plan.materialize_in(&ctx).unwrap();
    assert_eq!(last_exec_stream_value(), exec_rt.stream().testing_raw_handle_value());
    assert_eq!(last_exec_mr_value(), exec_rt.memory_resource().testing_raw_handle_value());
    assert_eq!(materialized.testing_execution_domain_stream_value(), exec_rt.stream().testing_raw_handle_value());
    assert_eq!(materialized.testing_execution_domain_mr_value(), exec_rt.memory_resource().testing_raw_handle_value());
    assert_eq!(reservation, device_allocated_bytes(&materialized));
}

fn test_projected_cross_join_one_sided_projection_does_not_order_unselected_side() {
    let _event_observation = crate::common::lock_event_observation();
    let left_rt = sample_runtime();
    let right_rt = sample_runtime();
    let left = cross_left_df(&left_rt);
    let right = cross_right_df(&right_rt);
    let plan = left
        .projected_cross_join_plan(
            &right,
            DataFrameCrossJoinOptions::new().left_projection([("left_val", "left_only")]),
        )
        .unwrap();

    reset_dependency_observation();
    let output_stream = left.testing_execution_domain_stream_value();
    let materialized = plan.materialize().unwrap();

    assert_eq!(materialized.testing_execution_domain_stream_value(), output_stream);
    assert_eq!(dependency_wait_call_count_for_stream(output_stream), 0);
}

fn test_projected_cross_join_plan_rejects_duplicate_output_names_at_creation() {
    let rt = sample_runtime();
    let left = cross_left_df(&rt);
    let right = cross_right_df(&rt);

    let err = left
        .projected_cross_join_plan(
            &right,
            DataFrameCrossJoinOptions::new().left_projection([("id", "dup")]).right_projection([("id", "dup")]),
        )
        .unwrap_err();

    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.code(), ErrorCode::DuplicateColumnName);
    assert_eq!(err.operation(), "dataframe::projected_cross_join_plan");
}

fn test_projected_cross_join_plan_rejects_zero_output_columns_explicitly() {
    let rt = sample_runtime();
    let left = cross_left_df(&rt);
    let right = cross_right_df(&rt);

    let err = left.projected_cross_join_plan(&right, DataFrameCrossJoinOptions::new()).unwrap_err();

    assert_merge_validation_error(
        &err,
        "dataframe::projected_cross_join_plan",
        "empty_cross_join_projection",
        "select at least one left or right output column",
    );
}

fn test_projected_cross_join_plan_rejects_row_count_overflow_before_execution() {
    let err = cudf::testing::join::trigger_projected_cross_join_output_row_overflow_for_test().unwrap_err();

    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(err.operation(), "dataframe::projected_cross_join_plan");
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("output_rows"));
}

fn test_projected_cross_join_plan_rejects_input_row_count_above_cudf_limit_even_with_zero_output() {
    let err = cudf::testing::join::trigger_projected_cross_join_input_row_limit_for_test().unwrap_err();

    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(err.operation(), "dataframe::projected_cross_join_plan");
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("left_rows"));
}

fn test_dataframe_join_indices_materialization_reservation_orders_mixed_domains() {
    let _event_observation = crate::common::lock_event_observation();
    let data_rt = sample_runtime();
    let indices_rt = sample_runtime();
    let left = DataFrame::from_series([
        ("id", data_rt.series(&[1i32, 2, 3, 4]).unwrap()),
        ("score", data_rt.series(&[10i32, 20, 30, 40]).unwrap()),
    ])
    .unwrap();
    let right = DataFrame::from_series([("id", data_rt.series(&[2i32, 4]).unwrap())]).unwrap();
    let indices_ctx =
        ExecutionOverrides::new().with_stream(indices_rt.stream()).with_memory_resource(indices_rt.memory_resource());
    let indices = left
        .with_execution(&indices_ctx)
        .left_semi_join_indices(&right, DataFrameJoinOptions::new().on(["id"]))
        .unwrap();
    let expected_exec_stream = left.testing_execution_domain_stream_value();
    let expected_exec_mr = left.testing_execution_domain_mr_value();

    reset_dependency_observation();
    reset_exec_observation();
    let reservation = left.materialize_join_indices_reservation_size_bytes(&indices, ["score"]).unwrap();
    assert_eq!(last_exec_stream_value(), expected_exec_stream);
    assert_eq!(last_exec_mr_value(), expected_exec_mr);
    assert_eq!(dependency_wait_call_count_for_stream(expected_exec_stream), 1);

    let materialized = left.materialize_join_indices(&indices, ["score"]).unwrap();
    assert_eq!(reservation, device_allocated_bytes(&materialized));
}

fn test_dataframe_inner_join_indices_materialize_projected_sides_with_duplicates() {
    let rt = sample_runtime();
    let left = DataFrame::from_series([
        ("id", rt.series(&[1i32, 2, 2, 3]).unwrap()),
        ("score", rt.series(&[10i32, 20, 21, 30]).unwrap()),
    ])
    .unwrap();
    let right = DataFrame::from_series([
        ("rid", rt.series(&[2i32, 2, 4]).unwrap()),
        ("value", rt.series(&[200i32, 201, 400]).unwrap()),
    ])
    .unwrap();

    let indices =
        left.inner_join_indices(&right, DataFrameJoinOptions::new().left_on(["id"]).right_on(["rid"])).unwrap();
    let left_rows = left.materialize_join_indices(indices.left(), ["score"]).unwrap();
    let right_rows = right.materialize_join_indices(indices.right(), ["value"]).unwrap();

    assert_eq!(indices.len(), 4);
    assert_eq!(
        DataFrameJoinPairIndices::reservation_size_bytes_for_len(indices.len()).unwrap(),
        indices.device_allocated_size_bytes().unwrap()
    );
    assert!(indices.device_allocated_size_bytes().unwrap() > 0);
    assert_eq!(left_rows.column_names(), vec!["score"]);
    assert_eq!(right_rows.column_names(), vec!["value"]);
    let mut rows: Vec<_> = i32_column(&left_rows, "score").into_iter().zip(i32_column(&right_rows, "value")).collect();
    rows.sort();
    assert_eq!(rows, vec![(20, 200), (20, 201), (21, 200), (21, 201)]);
}

fn test_dataframe_left_join_indices_nullify_unmatched_right_projection() {
    let rt = sample_runtime();
    let left = DataFrame::from_series([
        ("id", rt.series(&[1i32, 2, 3]).unwrap()),
        ("score", rt.series(&[10i32, 20, 30]).unwrap()),
    ])
    .unwrap();
    let right = DataFrame::from_series([("id", rt.series(&[2i32]).unwrap()), ("value", rt.series(&[200i32]).unwrap())])
        .unwrap();

    let indices = left.left_join_indices(&right, DataFrameJoinOptions::new().on(["id"])).unwrap();
    let left_rows = left.materialize_join_indices(indices.left(), ["id"]).unwrap();
    let right_rows = right.materialize_join_indices(indices.right(), ["value"]).unwrap();

    assert_eq!(indices.len(), 3);
    let mut rows: Vec<_> =
        i32_column(&left_rows, "id").into_iter().zip(nullable_i32_column(&right_rows, "value")).collect();
    rows.sort();
    assert_eq!(rows, vec![(1, None), (2, Some(200)), (3, None)]);
}

fn test_dataframe_full_join_indices_nullify_unmatched_sides() {
    let rt = sample_runtime();
    let left =
        DataFrame::from_series([("id", rt.series(&[1i32, 2]).unwrap()), ("score", rt.series(&[10i32, 20]).unwrap())])
            .unwrap();
    let right =
        DataFrame::from_series([("id", rt.series(&[2i32, 3]).unwrap()), ("value", rt.series(&[200i32, 300]).unwrap())])
            .unwrap();

    let indices = left.full_join_indices(&right, DataFrameJoinOptions::new().on(["id"])).unwrap();
    let left_rows = left.materialize_join_indices(indices.left(), ["score"]).unwrap();
    let right_rows = right.materialize_join_indices(indices.right(), ["value"]).unwrap();

    assert_eq!(indices.len(), 3);
    let mut rows: Vec<_> =
        nullable_i32_column(&left_rows, "score").into_iter().zip(nullable_i32_column(&right_rows, "value")).collect();
    rows.sort();
    assert_eq!(rows, vec![(None, Some(300)), (Some(10), None), (Some(20), Some(200))]);
}

fn test_dataframe_filtered_join_probe_state_materializes_without_second_size_probe() {
    let rt = sample_runtime();
    let build = DataFrame::from_series([("id", rt.series(&[2i32, 4]).unwrap())]).unwrap();
    let probe = DataFrame::from_series([
        ("pid", rt.series(&[1i32, 2, 2, 4]).unwrap()),
        ("score", rt.series(&[10i32, 20, 21, 40]).unwrap()),
    ])
    .unwrap();

    let filtered_join = build.build_filtered_join(["id"], DataFrameFilteredJoinOptions::new()).unwrap();
    assert!(filtered_join.device_allocated_size_bytes().unwrap() > 0);

    let semi_state = filtered_join.begin_left_semi_probe(&probe, ["pid"]).unwrap();
    assert_eq!(semi_state.output_size(), 3);
    assert!(semi_state.device_allocated_size_bytes().unwrap() > 0);
    assert_eq!(
        semi_state.probe_gather_map_reservation_size_bytes().unwrap(),
        DataFrameJoinIndices::reservation_size_bytes_for_len(semi_state.output_size()).unwrap()
    );
    let semi_indices = semi_state.materialize_indices().unwrap();
    assert_eq!(semi_indices.len(), 3);
    let semi = filtered_join.materialize_probe(&probe, &semi_indices, ["score"]).unwrap();
    let mut semi_scores = i32_column(&semi, "score");
    semi_scores.sort();
    assert_eq!(semi_scores, vec![20, 21, 40]);

    let anti_state = filtered_join.begin_left_anti_probe(&probe, ["pid"]).unwrap();
    assert_eq!(anti_state.output_size(), 1);
    assert!(anti_state.device_allocated_size_bytes().unwrap() > 0);
    let anti_indices = anti_state.materialize_indices().unwrap();
    let anti = filtered_join.materialize_probe(&probe, &anti_indices, ["score"]).unwrap();
    assert_eq!(i32_column(&anti, "score"), vec![10]);
}

fn test_filtered_join_fixed_width_metadata_preflight_matches_built_owner() {
    let build = crate::common::dataframe_from_columns(
        vec!["leftKey".to_string(), "rightKey".to_string()],
        vec![
            column_from_slice_nullable(&[2i64, 4, 8], &[true, true, true]).unwrap(),
            column_from_slice_nullable(&[3i64, 5, 9], &[true, true, true]).unwrap(),
        ],
    )
    .unwrap();
    let options = DataFrameFilteredJoinOptions::new().null_equality(cudf::NullEquality::Unequal);
    let metadata_bytes = FilteredJoin::pre_build_reservation_size_bytes_for_fixed_width_metadata(
        &[TypeId::Int64, TypeId::Int64],
        build.num_rows(),
        FilteredJoinOptions::new().null_equality(cudf::NullEquality::Unequal),
    )
    .unwrap();
    let frame_bytes = build.filtered_join_pre_build_reservation_size_bytes(["leftKey", "rightKey"], options).unwrap();
    let owner = build.build_filtered_join(["leftKey", "rightKey"], options).unwrap();

    assert_eq!(metadata_bytes, frame_bytes);
    assert_eq!(metadata_bytes, owner.device_allocated_size_bytes().unwrap());
}

fn test_filtered_join_string_metadata_and_probe_preflights_match_built_owners() {
    let rt = sample_runtime();
    let build = DataFrame::from_series([("id", rt.series(&["a", "cc"]).unwrap())]).unwrap();
    let probe = DataFrame::from_series([("id", rt.series(&["a", "b", "cc"]).unwrap())]).unwrap();
    let options = DataFrameFilteredJoinOptions::new().null_equality(cudf::NullEquality::Unequal);
    let metadata = filtered_join_allocation_preflight_for_metadata(
        &[HashJoinKeyMetadata::string(3, HashJoinStringOffsetWidth::I32, false)],
        build.num_rows() as u64,
        FilteredJoinOptions::new().null_equality(cudf::NullEquality::Unequal),
    )
    .unwrap();
    let owner = build.build_filtered_join(["id"], options).unwrap();

    assert_eq!(metadata.retained_build_bytes(), owner.device_allocated_size_bytes().unwrap());
    let probe_state = owner.begin_left_semi_probe(&probe, ["id"]).unwrap();
    let probe_preflight =
        filtered_join_probe_allocation_preflight(probe.num_rows() as u64, probe_state.output_size() as u64).unwrap();
    assert_eq!(probe_preflight.retained_probe_state_bytes(), probe_state.device_allocated_size_bytes().unwrap());
    assert_eq!(probe_preflight.gather_map_bytes(), probe_state.probe_gather_map_reservation_size_bytes().unwrap());

    let nullable_build = crate::common::dataframe_from_columns(
        vec!["id".to_string()],
        vec![column_from_strs_nullable(&["a", "", "cc"], &[true, false, true]).unwrap()],
    )
    .unwrap();
    let nullable_metadata = filtered_join_allocation_preflight_for_metadata(
        &[HashJoinKeyMetadata::string(3, HashJoinStringOffsetWidth::I32, true)],
        nullable_build.num_rows() as u64,
        FilteredJoinOptions::new().null_equality(cudf::NullEquality::Equal),
    )
    .unwrap();
    let nullable_owner = nullable_build
        .build_filtered_join(["id"], DataFrameFilteredJoinOptions::new().null_equality(cudf::NullEquality::Equal))
        .unwrap();
    assert_eq!(nullable_metadata.retained_build_bytes(), nullable_owner.device_allocated_size_bytes().unwrap());
    assert_eq!(nullable_metadata.temporary_row_validity_workspace_bytes(), 0);
}

fn test_dataframe_hash_join_reuses_build_for_projected_inner_probes_with_duplicates() {
    let rt = sample_runtime();
    let build = DataFrame::from_series([
        ("id", rt.series(&[2i32, 2, 4]).unwrap()),
        ("value", rt.series(&[200i32, 201, 400]).unwrap()),
    ])
    .unwrap();
    let first_probe = DataFrame::from_series([
        ("pid", rt.series(&[1i32, 2, 2, 3]).unwrap()),
        ("score", rt.series(&[10i32, 20, 21, 30]).unwrap()),
    ])
    .unwrap();
    let second_probe =
        DataFrame::from_series([("pid", rt.series(&[4i32, 5]).unwrap()), ("score", rt.series(&[40i32, 50]).unwrap())])
            .unwrap();

    let hash_join = build.build_hash_join(["id"], DataFrameHashJoinOptions::new()).unwrap();
    assert!(hash_join.device_allocated_size_bytes().unwrap() > 0);
    let first_output_size = hash_join.inner_probe_size(&first_probe, ["pid"]).unwrap();
    let first_reservation_size =
        DataFrameHashJoinPairIndices::reservation_size_bytes_for_len(first_output_size).unwrap();
    assert_eq!(hash_join.probe_gather_map_reservation_size_bytes(first_output_size).unwrap(), first_reservation_size);
    let first_indices = hash_join.inner_probe(&first_probe, ["pid"]).unwrap();
    assert_eq!(first_reservation_size, first_indices.device_allocated_size_bytes().unwrap());
    assert!(first_indices.device_allocated_size_bytes().unwrap() > 0);
    let probe_output_reservation =
        first_probe.materialize_join_indices_reservation_size_bytes(first_indices.probe(), ["score"]).unwrap();
    let build_output_reservation =
        hash_join.build().materialize_join_indices_reservation_size_bytes(first_indices.build(), ["value"]).unwrap();
    let first = hash_join.materialize(&first_probe, &first_indices, ["score"], ["value"]).unwrap();
    let second_indices = hash_join.inner_probe(&second_probe, ["pid"]).unwrap();
    let second = hash_join.materialize(&second_probe, &second_indices, ["score"], ["value"]).unwrap();

    assert_eq!(probe_output_reservation, device_allocated_bytes(&first.probe));
    assert_eq!(build_output_reservation, device_allocated_bytes(&first.build));
    assert_eq!(hash_join.build_keys(), &["id".to_string()]);
    assert_eq!(hash_join.inner_probe_size(&first_probe, ["pid"]).unwrap(), first_indices.len());
    assert_eq!(hash_join.inner_probe_size(&second_probe, ["pid"]).unwrap(), second_indices.len());
    assert_eq!(first_indices.len(), 4);
    let mut first_rows: Vec<_> =
        i32_column(&first.probe, "score").into_iter().zip(i32_column(&first.build, "value")).collect();
    first_rows.sort();
    assert_eq!(first_rows, vec![(20, 200), (20, 201), (21, 200), (21, 201)]);
    assert_eq!(second_indices.len(), 1);
    assert_eq!(i32_column(&second.probe, "score"), vec![40]);
    assert_eq!(i32_column(&second.build, "value"), vec![400]);
}

fn test_dataframe_mixed_join_size_data_materializes_inner_and_left_without_recomputing_size() {
    let rt = sample_runtime();
    let left = DataFrame::from_series([
        ("id", rt.series(&[1i32, 1, 2, 2, 4]).unwrap()),
        ("left_cond", rt.series(&[5i32, 7, 3, 9, 1]).unwrap()),
        ("score", rt.series(&[50i32, 70, 30, 90, 10]).unwrap()),
    ])
    .unwrap();
    let right = DataFrame::from_series([
        ("id", rt.series(&[1i32, 1, 2, 2, 3]).unwrap()),
        ("right_cond", rt.series(&[6i32, 8, 4, 7, 10]).unwrap()),
        ("value", rt.series(&[60i32, 80, 40, 70, 100]).unwrap()),
    ])
    .unwrap();
    let options = DataFrameMixedJoinOptions::new()
        .on(["id"])
        .left_predicate_columns(["left_cond"])
        .right_predicate_columns(["right_cond"]);
    let predicate = left_less_than_right_predicate();

    let inner_state = left.mixed_inner_join_size_data(&right, &predicate, options.clone()).unwrap();
    assert_eq!(inner_state.output_size(), 5);
    assert!(inner_state.device_allocated_size_bytes().unwrap() > 0);
    assert_eq!(
        inner_state.gather_map_reservation_size_bytes().unwrap(),
        DataFrameJoinPairIndices::reservation_size_bytes_for_len(inner_state.output_size()).unwrap()
    );
    let inner_gather_reservation = inner_state.gather_map_reservation_size_bytes().unwrap();
    let inner_indices = inner_state.materialize_indices().unwrap();
    assert_eq!(inner_indices.device_allocated_size_bytes().unwrap(), inner_gather_reservation);
    assert_eq!(sorted_pair_indices(&inner_indices), vec![(0, 0), (0, 1), (1, 1), (2, 2), (2, 3)]);

    let left_state = left.mixed_left_join_size_data(&right, &predicate, options).unwrap();
    assert_eq!(left_state.output_size(), 7);
    assert!(left_state.device_allocated_size_bytes().unwrap() > 0);
    let left_indices = left_state.materialize_indices().unwrap();
    assert_eq!(
        sorted_pair_indices(&left_indices),
        vec![(0, 0), (0, 1), (1, 1), (2, 2), (2, 3), (3, i32::MIN), (4, i32::MIN)]
    );
}

fn test_dataframe_hash_join_left_and_full_probes_nullify_unmatched_projection() {
    let rt = sample_runtime();
    let build =
        DataFrame::from_series([("id", rt.series(&[2i32, 3]).unwrap()), ("value", rt.series(&[200i32, 300]).unwrap())])
            .unwrap();
    let probe =
        DataFrame::from_series([("pid", rt.series(&[1i32, 2]).unwrap()), ("score", rt.series(&[10i32, 20]).unwrap())])
            .unwrap();

    let hash_join = build.build_hash_join(["id"], DataFrameHashJoinOptions::new()).unwrap();
    let left_indices = hash_join.left_probe(&probe, ["pid"]).unwrap();
    let left_probe = hash_join.materialize_probe(&probe, &left_indices, ["score"]).unwrap();
    let left_build = hash_join.materialize_build(&left_indices, ["value"]).unwrap();
    let full_indices = hash_join.full_probe(&probe, ["pid"]).unwrap();
    let full_probe = hash_join.materialize_probe(&probe, &full_indices, ["score"]).unwrap();
    let full_build = hash_join.materialize_build(&full_indices, ["value"]).unwrap();

    assert_eq!(hash_join.left_probe_size(&probe, ["pid"]).unwrap(), left_indices.len());
    assert_eq!(hash_join.full_probe_size(&probe, ["pid"]).unwrap(), full_indices.len());

    let mut left_rows: Vec<_> =
        i32_column(&left_probe, "score").into_iter().zip(nullable_i32_column(&left_build, "value")).collect();
    left_rows.sort();
    assert_eq!(left_rows, vec![(10, None), (20, Some(200))]);

    let mut full_rows: Vec<_> =
        nullable_i32_column(&full_probe, "score").into_iter().zip(nullable_i32_column(&full_build, "value")).collect();
    full_rows.sort();
    assert_eq!(full_rows, vec![(None, Some(300)), (Some(10), None), (Some(20), Some(200))]);
}

fn test_dataframe_left_semi_join_honors_null_equality() {
    let left_ids = column_from_slice_nullable(&[1i32, 0, 2], &[true, false, true]).unwrap();
    let scores = column_from_slice(&[10i32, 20, 30]).unwrap();
    let right_ids = column_from_slice_nullable(&[0i32, 3], &[false, true]).unwrap();
    let left =
        crate::common::dataframe_from_columns(vec!["id".to_string(), "score".to_string()], vec![left_ids, scores])
            .unwrap();
    let right = crate::common::dataframe_from_columns(vec!["id".to_string()], vec![right_ids]).unwrap();

    let nulls_equal = left
        .left_semi_join(&right, DataFrameJoinOptions::new().on(["id"]).null_equality(cudf::NullEquality::Equal))
        .unwrap();
    let nulls_unequal = left
        .left_semi_join(&right, DataFrameJoinOptions::new().on(["id"]).null_equality(cudf::NullEquality::Unequal))
        .unwrap();

    assert_eq!(nullable_i32_column(&nulls_equal, "id"), vec![None]);
    assert_eq!(i32_column(&nulls_equal, "score"), vec![20]);
    assert_eq!(nulls_unequal.num_rows(), 0);
}

fn test_dataframe_left_semi_join_indices_with_execution_uses_borrowed_handles() {
    let _event_observation = crate::common::lock_event_observation();
    let data_rt = sample_runtime();
    let query_rt = sample_runtime();
    let left = DataFrame::from_series([
        ("id", data_rt.series(&[1i32, 2, 3, 4]).unwrap()),
        ("score", data_rt.series(&[10i32, 20, 30, 40]).unwrap()),
    ])
    .unwrap();
    let right = DataFrame::from_series([("id", data_rt.series(&[2i32, 4]).unwrap())]).unwrap();
    let ctx = ExecutionOverrides::new().with_stream(query_rt.stream()).with_memory_resource(query_rt.memory_resource());

    reset_exec_observation();
    let indices =
        left.with_execution(&ctx).left_semi_join_indices(&right, DataFrameJoinOptions::new().on(["id"])).unwrap();

    assert_eq!(last_exec_stream_value(), query_rt.stream().testing_raw_handle_value());
    assert_eq!(last_exec_mr_value(), query_rt.memory_resource().testing_raw_handle_value());
    assert_eq!(indices.placement().device, query_rt.device());
}

fn test_dataframe_inner_join_indices_with_execution_uses_borrowed_handles() {
    let _event_observation = crate::common::lock_event_observation();
    let data_rt = sample_runtime();
    let query_rt = sample_runtime();
    let left = DataFrame::from_series([
        ("id", data_rt.series(&[1i32, 2, 3, 4]).unwrap()),
        ("score", data_rt.series(&[10i32, 20, 30, 40]).unwrap()),
    ])
    .unwrap();
    let right = DataFrame::from_series([("id", data_rt.series(&[2i32, 4]).unwrap())]).unwrap();
    let ctx = ExecutionOverrides::new().with_stream(query_rt.stream()).with_memory_resource(query_rt.memory_resource());

    reset_exec_observation();
    let indices = left.with_execution(&ctx).inner_join_indices(&right, DataFrameJoinOptions::new().on(["id"])).unwrap();

    assert_eq!(last_exec_stream_value(), query_rt.stream().testing_raw_handle_value());
    assert_eq!(last_exec_mr_value(), query_rt.memory_resource().testing_raw_handle_value());
    assert_eq!(indices.placement().device, query_rt.device());
}

fn test_dataframe_left_semi_join_rejects_missing_join_keys() {
    let left = left_df();
    let right = right_df();

    let err = left.left_semi_join(&right, DataFrameJoinOptions::new()).unwrap_err();

    assert_merge_validation_error(
        &err,
        "dataframe::left_semi_join_indices",
        "missing_join_keys",
        "set on() or left_on() + right_on()",
    );
}

fn test_dataframe_left_semi_join_rejects_ambiguous_join_key_configuration() {
    let left = left_df();
    let right = right_df();

    let err = left.left_semi_join(&right, DataFrameJoinOptions::new().on(["id"]).left_on(["id"])).unwrap_err();

    assert_merge_validation_error(
        &err,
        "dataframe::left_semi_join_indices",
        "ambiguous_join_key_configuration",
        "use on() or left_on() + right_on(), not both",
    );
}

fn test_dataframe_left_anti_join_rejects_left_on_without_right_on() {
    let left = left_df();
    let right = right_df();

    let err = left.left_anti_join(&right, DataFrameJoinOptions::new().left_on(["id"])).unwrap_err();

    assert_merge_validation_error(
        &err,
        "dataframe::left_anti_join_indices",
        "left_on_without_right_on",
        "set right_on() or use on()",
    );
}

fn test_dataframe_left_semi_join_indices_rejects_key_count_mismatch() {
    let left = left_df_asym();
    let right = right_df_asym();

    let err = left
        .left_semi_join_indices(
            &right,
            DataFrameJoinOptions::new().left_on(["left_id"]).right_on(["right_id", "value"]),
        )
        .unwrap_err();

    assert_merge_validation_error(
        &err,
        "dataframe::left_semi_join_indices",
        "join_key_count_mismatch",
        "set left_on() and right_on() with the same number of key columns",
    );
    assert_eq!(error_context_value(&err, "left_on_count").as_deref(), Some("1"));
    assert_eq!(error_context_value(&err, "right_on_count").as_deref(), Some("2"));
}

fn test_dataframe_materialize_join_indices_reanchors_unknown_column() {
    let left = left_df();
    let right = right_df();

    let indices = left.left_semi_join_indices(&right, DataFrameJoinOptions::new().on(["id"])).unwrap();
    let err = left.materialize_join_indices(&indices, ["missing_payload"]).unwrap_err();

    assert_missing_column_error(&err, "missing_payload");
    assert_eq!(err.operation(), "dataframe::materialize_join_indices");
}

fn test_dataframe_left_semi_join_empty_result_preserves_schema() {
    let rt = sample_runtime();
    let left =
        DataFrame::from_series([("id", rt.series(&[1i32, 2]).unwrap()), ("score", rt.series(&[10i32, 20]).unwrap())])
            .unwrap();
    let right = DataFrame::from_series([("id", rt.series(&[3i32, 4]).unwrap())]).unwrap();

    let indices = left.left_semi_join_indices(&right, DataFrameJoinOptions::new().on(["id"])).unwrap();
    let result = left.left_semi_join(&right, DataFrameJoinOptions::new().on(["id"])).unwrap();

    assert!(indices.is_empty());
    assert_eq!(result.column_names(), vec!["id", "score"]);
    assert_eq!(result.num_rows(), 0);
    assert!(i32_column(&result, "id").is_empty());
    assert!(i32_column(&result, "score").is_empty());
}

fn test_dataframe_left_anti_join_empty_result_preserves_schema() {
    let rt = sample_runtime();
    let left =
        DataFrame::from_series([("id", rt.series(&[1i32, 2]).unwrap()), ("score", rt.series(&[10i32, 20]).unwrap())])
            .unwrap();
    let right = DataFrame::from_series([("id", rt.series(&[1i32, 2]).unwrap())]).unwrap();

    let indices = left.left_anti_join_indices(&right, DataFrameJoinOptions::new().on(["id"])).unwrap();
    let result = left.left_anti_join(&right, DataFrameJoinOptions::new().on(["id"])).unwrap();

    assert!(indices.is_empty());
    assert_eq!(result.column_names(), vec!["id", "score"]);
    assert_eq!(result.num_rows(), 0);
    assert!(i32_column(&result, "id").is_empty());
    assert!(i32_column(&result, "score").is_empty());
}

// ---------------------------------------------------------------------------
// MergeOptions builder
// ---------------------------------------------------------------------------

fn test_merge_options_defaults() {
    let opts = MergeOptions::new();
    assert_eq!(opts.join_type(), JoinType::Inner);
    assert_eq!(opts.suffix_left(), "_left");
    assert_eq!(opts.suffix_right(), "_right");
}

fn test_merge_options_builder_on() {
    let opts = MergeOptions::new().on(["id"]);
    let on = opts.on_keys().unwrap();
    assert_eq!(on.len(), 1);
    assert_eq!(on[0], "id");
}

fn test_merge_options_builder_left_right_on() {
    let opts = MergeOptions::new().left_on(["left_id"]).right_on(["right_id"]);
    let lo = opts.left_on_keys().unwrap();
    let ro = opts.right_on_keys().unwrap();
    assert_eq!(lo.len(), 1);
    assert_eq!(lo[0], "left_id");
    assert_eq!(ro.len(), 1);
    assert_eq!(ro[0], "right_id");
}

fn test_merge_options_builder_how() {
    assert_eq!(MergeOptions::new().how(JoinType::Left).join_type(), JoinType::Left);
    assert_eq!(MergeOptions::new().how_inner().join_type(), JoinType::Inner);
    assert_eq!(MergeOptions::new().how_left().join_type(), JoinType::Left);
    assert_eq!(MergeOptions::new().how_full().join_type(), JoinType::Full);
}

fn test_merge_options_builder_suffixes() {
    let opts = MergeOptions::new().suffixes("_x", "_y");
    assert_eq!(opts.suffix_left(), "_x");
    assert_eq!(opts.suffix_right(), "_y");
}

// ---------------------------------------------------------------------------
// Inner merge — same key name (on)
// ---------------------------------------------------------------------------

fn test_inner_merge_same_key() {
    let left = left_df();
    let right = right_df();

    let result = left.merge(&right, MergeOptions::new().on(["id"])).unwrap();

    // Inner: matches on 2 and 3 → 2 rows.
    assert_eq!(result.num_rows(), 2);
    // Schema: shared key + left payload + right payload.
    assert_eq!(result.column_names(), vec!["id", "score", "value"]);

    let mut rows: Vec<_> = i32_column(&result, "id")
        .into_iter()
        .zip(i32_column(&result, "score"))
        .zip(i32_column(&result, "value"))
        .map(|((id, score), value)| (id, score, value))
        .collect();
    rows.sort();
    assert_eq!(rows, vec![(2, 20, 200), (3, 30, 300)]);
}

fn test_merge_succeeds_with_mixed_domains_and_uses_left_provenance() {
    let _event_observation = crate::common::lock_event_observation();
    let left_rt = sample_runtime();
    let right_rt = sample_runtime();
    let left = DataFrame::from_series([
        ("id", left_rt.series(&[1i32, 2, 3]).unwrap()),
        ("score", left_rt.series(&[10i32, 20, 30]).unwrap()),
    ])
    .unwrap();
    let right = DataFrame::from_series([
        ("id", right_rt.series(&[2i32, 3, 4]).unwrap()),
        ("value", right_rt.series(&[200i32, 300, 400]).unwrap()),
    ])
    .unwrap();
    let expected_exec_stream = left.testing_execution_domain_stream_value();
    let expected_exec_mr = left.testing_execution_domain_mr_value();
    crate::common::reset_dependency_observation();
    reset_exec_observation();
    let result = left.merge(&right, MergeOptions::new().on(["id"])).unwrap();

    assert_eq!(result.testing_execution_domain_id(), left.testing_execution_domain_id());
    assert_eq!(result.testing_execution_domain_mr_value(), expected_exec_mr);
    assert_eq!(last_exec_stream_value(), expected_exec_stream);
    assert_eq!(last_exec_mr_value(), expected_exec_mr);
    // DataFrame guarantees one retained provenance bundle for all columns, so
    // one dependency from the right frame's stream orders both right keys and
    // payload before the left/output stream.
    assert_eq!(crate::common::dependency_wait_call_count_for_stream(expected_exec_stream), 1);

    reset_sync_observation();
    let rows: Vec<_> = i32_column(&result, "id")
        .into_iter()
        .zip(i32_column(&result, "score"))
        .zip(i32_column(&result, "value"))
        .map(|((id, score), value)| (id, score, value))
        .collect();
    assert_eq!(rows, vec![(2, 20, 200), (3, 30, 300)]);
    assert_eq!(last_sync_stream_value(), expected_exec_stream);
}

// ---------------------------------------------------------------------------
// Left merge — different key names (left_on / right_on)
// ---------------------------------------------------------------------------

fn test_left_merge_asymmetric_keys() {
    let left = left_df_asym();
    let right = right_df_asym();

    let result =
        left.merge(&right, MergeOptions::new().how_left().left_on(["left_id"]).right_on(["right_id"])).unwrap();

    // Left: all 3 left rows preserved.
    assert_eq!(result.num_rows(), 3);
    // With left_on/right_on, both key columns are preserved in MVP.
    assert_eq!(result.column_names(), vec!["left_id", "score", "right_id", "value"]);

    let mut rows: Vec<_> = nullable_i32_column(&result, "left_id")
        .into_iter()
        .zip(nullable_i32_column(&result, "score"))
        .zip(nullable_i32_column(&result, "right_id"))
        .zip(nullable_i32_column(&result, "value"))
        .map(|(((left_id, score), right_id), value)| (left_id, score, right_id, value))
        .collect();
    rows.sort();
    assert_eq!(
        rows,
        vec![
            (Some(1), Some(10), None, None),
            (Some(2), Some(20), Some(2), Some(200)),
            (Some(3), Some(30), Some(3), Some(300)),
        ]
    );
}

// ---------------------------------------------------------------------------
// Suffix handling for overlapping payload names
// ---------------------------------------------------------------------------

fn test_merge_suffix_overlapping_payload() {
    // Both sides have "score" as payload.
    let left = left_df();
    let right_ids = column_from_slice(&[2i32, 3, 4]).unwrap();
    let right_scores = column_from_slice(&[200i32, 300, 400]).unwrap();
    let right = crate::common::dataframe_from_columns(
        vec!["id".to_string(), "score".to_string()],
        vec![right_ids, right_scores],
    )
    .unwrap();

    let result = left.merge(&right, MergeOptions::new().on(["id"])).unwrap();

    // Default suffixes: _left, _right for conflicting payloads.
    assert_eq!(result.column_names(), vec!["id", "score_left", "score_right"]);

    let mut rows: Vec<_> = i32_column(&result, "id")
        .into_iter()
        .zip(i32_column(&result, "score_left"))
        .zip(i32_column(&result, "score_right"))
        .map(|((id, left_score), right_score)| (id, left_score, right_score))
        .collect();
    rows.sort();
    assert_eq!(rows, vec![(2, 20, 200), (3, 30, 300)]);
}

fn test_merge_custom_suffixes() {
    let left = left_df();
    let right_ids = column_from_slice(&[2i32, 3, 4]).unwrap();
    let right_scores = column_from_slice(&[200i32, 300, 400]).unwrap();
    let right = crate::common::dataframe_from_columns(
        vec!["id".to_string(), "score".to_string()],
        vec![right_ids, right_scores],
    )
    .unwrap();

    let result = left.merge(&right, MergeOptions::new().on(["id"]).suffixes("_x", "_y")).unwrap();

    assert_eq!(result.column_names(), vec!["id", "score_x", "score_y"]);
}

fn test_merge_no_suffix_when_no_conflict() {
    let left = left_df();
    let right = right_df();

    let result = left.merge(&right, MergeOptions::new().on(["id"])).unwrap();

    // No conflict: "score" and "value" are distinct.
    assert_eq!(result.column_names(), vec!["id", "score", "value"]);
}

// ---------------------------------------------------------------------------
// Full merge
// ---------------------------------------------------------------------------

fn test_full_merge() {
    let left = left_df();
    let right = right_df();

    let result = left.merge(&right, MergeOptions::new().how_full().on(["id"])).unwrap();

    // Full: keys 1,2,3,4 → 4 rows.
    assert_eq!(result.num_rows(), 4);
    assert_eq!(result.column_names(), vec!["id", "score", "value"]);

    let mut rows: Vec<_> = nullable_i32_column(&result, "id")
        .into_iter()
        .zip(nullable_i32_column(&result, "score"))
        .zip(nullable_i32_column(&result, "value"))
        .map(|((id, score), value)| (id, score, value))
        .collect();
    rows.sort();
    assert_eq!(
        rows,
        vec![
            (Some(1), Some(10), None),
            (Some(2), Some(20), Some(200)),
            (Some(3), Some(30), Some(300)),
            (Some(4), None, Some(400)),
        ]
    );
}

fn test_full_merge_succeeds_with_mixed_domains_and_reassembles_on_left_provenance() {
    let _event_observation = crate::common::lock_event_observation();
    let left_rt = sample_runtime();
    let right_rt = sample_runtime();
    let left = DataFrame::from_series([
        ("id", left_rt.series(&[1i32, 2, 3]).unwrap()),
        ("score", left_rt.series(&[10i32, 20, 30]).unwrap()),
    ])
    .unwrap();
    let right = DataFrame::from_series([
        ("id", right_rt.series(&[2i32, 3, 4]).unwrap()),
        ("value", right_rt.series(&[200i32, 300, 400]).unwrap()),
    ])
    .unwrap();
    let expected_exec_stream = left.testing_execution_domain_stream_value();
    let expected_exec_mr = left.testing_execution_domain_mr_value();
    crate::common::reset_dependency_observation();
    reset_exec_observation();
    let result = left.merge(&right, MergeOptions::new().how_full().on(["id"])).unwrap();

    assert_eq!(result.testing_execution_domain_id(), left.testing_execution_domain_id());
    assert_eq!(result.testing_execution_domain_mr_value(), expected_exec_mr);
    assert_eq!(last_exec_stream_value(), expected_exec_stream);
    assert_eq!(last_exec_mr_value(), expected_exec_mr);
    // DataFrame-level merge can order the whole right frame once because all
    // columns in a DataFrame share one retained provenance bundle.
    assert_eq!(crate::common::dependency_wait_call_count_for_stream(expected_exec_stream), 1);
    assert_eq!(result.series("id").unwrap().as_column().testing_execution_domain_stream_value(), expected_exec_stream);
    assert_eq!(
        result.series("score").unwrap().as_column().testing_execution_domain_stream_value(),
        expected_exec_stream
    );
    assert_eq!(
        result.series("value").unwrap().as_column().testing_execution_domain_stream_value(),
        expected_exec_stream
    );

    // Verify materialized values from the full merge result.
    reset_sync_observation();
    let mut ids = result.series("id").unwrap().to_optional_vec::<i32>().unwrap();
    ids.sort();
    assert_eq!(ids, vec![Some(1), Some(2), Some(3), Some(4)]);
    assert_eq!(last_sync_stream_value(), expected_exec_stream);

    reset_sync_observation();
    let mut rows: Vec<_> = nullable_i32_column(&result, "id")
        .into_iter()
        .zip(nullable_i32_column(&result, "score"))
        .zip(nullable_i32_column(&result, "value"))
        .map(|((id, score), value)| (id, score, value))
        .collect();
    rows.sort();
    assert_eq!(
        rows,
        vec![
            (Some(1), Some(10), None),
            (Some(2), Some(20), Some(200)),
            (Some(3), Some(30), Some(300)),
            (Some(4), None, Some(400)),
        ]
    );
    assert_eq!(last_sync_stream_value(), expected_exec_stream);
}

fn test_full_merge_in_context_succeeds_on_mixed_domains_and_reassembles_on_context_resources() {
    let _event_observation = crate::common::lock_event_observation();
    let left_rt = sample_runtime();
    let right_rt = sample_runtime();
    let target_rt = sample_runtime();
    let left = DataFrame::from_series([
        ("id", left_rt.series(&[1i32, 2, 3]).unwrap()),
        ("score", left_rt.series(&[10i32, 20, 30]).unwrap()),
    ])
    .unwrap();
    let right = DataFrame::from_series([
        ("id", right_rt.series(&[2i32, 3, 4]).unwrap()),
        ("value", right_rt.series(&[200i32, 300, 400]).unwrap()),
    ])
    .unwrap();
    let ctx =
        ExecutionOverrides::new().with_stream(target_rt.stream()).with_memory_resource(target_rt.memory_resource());
    let expected_exec_stream = target_rt.stream().testing_raw_handle_value();
    let expected_exec_mr = target_rt.memory_resource().testing_raw_handle_value();
    crate::common::reset_dependency_observation();
    reset_exec_observation();
    let result = left.with_execution(&ctx).merge(&right, MergeOptions::new().how_full().on(["id"])).unwrap();

    assert_eq!(result.testing_execution_domain_id(), left.testing_execution_domain_id());
    assert_eq!(result.testing_execution_domain_stream_value(), expected_exec_stream);
    assert_eq!(result.testing_execution_domain_mr_value(), expected_exec_mr);
    assert_eq!(last_exec_stream_value(), expected_exec_stream);
    assert_eq!(last_exec_mr_value(), expected_exec_mr);
    assert_eq!(crate::common::dependency_wait_call_count_for_stream(expected_exec_stream), 2);
    assert_eq!(result.series("id").unwrap().as_column().testing_execution_domain_stream_value(), expected_exec_stream);
    assert_eq!(
        result.series("score").unwrap().as_column().testing_execution_domain_stream_value(),
        expected_exec_stream
    );
    assert_eq!(
        result.series("value").unwrap().as_column().testing_execution_domain_stream_value(),
        expected_exec_stream
    );

    // Verify materialized values from the full merge result.
    reset_sync_observation();
    let mut ids = result.series("id").unwrap().to_optional_vec::<i32>().unwrap();
    ids.sort();
    assert_eq!(ids, vec![Some(1), Some(2), Some(3), Some(4)]);
    assert_eq!(last_sync_stream_value(), expected_exec_stream);

    reset_sync_observation();
    let mut rows: Vec<_> = nullable_i32_column(&result, "id")
        .into_iter()
        .zip(nullable_i32_column(&result, "score"))
        .zip(nullable_i32_column(&result, "value"))
        .map(|((id, score), value)| (id, score, value))
        .collect();
    rows.sort();
    assert_eq!(
        rows,
        vec![
            (Some(1), Some(10), None),
            (Some(2), Some(20), Some(200)),
            (Some(3), Some(30), Some(300)),
            (Some(4), None, Some(400)),
        ]
    );
    assert_eq!(last_sync_stream_value(), expected_exec_stream);
}

fn test_full_merge_asymmetric_keys_preserves_separate_key_columns() {
    let left = left_df_asym();
    let right = right_df_asym();

    let result =
        left.merge(&right, MergeOptions::new().how_full().left_on(["left_id"]).right_on(["right_id"])).unwrap();

    assert_eq!(result.num_rows(), 4);
    assert_eq!(result.column_names(), vec!["left_id", "score", "right_id", "value"]);

    let mut rows: Vec<_> = nullable_i32_column(&result, "left_id")
        .into_iter()
        .zip(nullable_i32_column(&result, "score"))
        .zip(nullable_i32_column(&result, "right_id"))
        .zip(nullable_i32_column(&result, "value"))
        .map(|(((left_id, score), right_id), value)| (left_id, score, right_id, value))
        .collect();
    rows.sort();

    assert_eq!(
        rows,
        vec![
            (None, None, Some(4), Some(400)),
            (Some(1), Some(10), None, None),
            (Some(2), Some(20), Some(2), Some(200)),
            (Some(3), Some(30), Some(3), Some(300)),
        ]
    );
}

// ---------------------------------------------------------------------------
// Invalid key errors
// ---------------------------------------------------------------------------

fn test_merge_missing_on_key_in_left() {
    let left = left_df();
    let right = right_df();

    let err = left.merge(&right, MergeOptions::new().on(["nonexistent"])).unwrap_err();
    assert_missing_column_error(&err, "nonexistent");
    assert_eq!(err.operation(), "dataframe::merge");
}

fn test_merge_missing_on_key_in_right() {
    let left = left_df();
    let right = right_df();

    let err = left.merge(&right, MergeOptions::new().on(["score"])).unwrap_err();
    assert_missing_column_error(&err, "score");
}

fn test_merge_left_on_without_right_on() {
    let left = left_df();
    let right = right_df();

    let err = left.merge(&right, MergeOptions::new().left_on(["id"])).unwrap_err();
    assert_merge_validation_error(&err, "dataframe::merge", "left_on_without_right_on", "set right_on() or use on()");
}

fn test_merge_right_on_without_left_on() {
    let left = left_df();
    let right = right_df();

    let err = left.merge(&right, MergeOptions::new().right_on(["id"])).unwrap_err();
    assert_merge_validation_error(&err, "dataframe::merge", "right_on_without_left_on", "set left_on() or use on()");
}

fn test_merge_left_on_right_on_length_mismatch() {
    let left = left_df_asym();
    let right = right_df_asym();

    let err = left.merge(&right, MergeOptions::new().left_on(["left_id"]).right_on(["right_id", "value"])).unwrap_err();
    assert_merge_validation_error(
        &err,
        "dataframe::merge",
        "join_key_count_mismatch",
        "set left_on() and right_on() with the same number of key columns",
    );
    assert_eq!(error_context_value(&err, "left_on_count").as_deref(), Some("1"));
    assert_eq!(error_context_value(&err, "right_on_count").as_deref(), Some("2"));
}

fn test_merge_no_keys_specified() {
    let left = left_df();
    let right = right_df();

    let err = left.merge(&right, MergeOptions::new()).unwrap_err();
    assert_merge_validation_error(&err, "dataframe::merge", "missing_join_keys", "set on() or left_on() + right_on()");
}

fn test_merge_in_context_no_keys_specified() {
    let left = left_df();
    let right = right_df();
    let rt = sample_runtime();
    let ctx = ExecutionOverrides::new().with_stream(rt.stream()).with_memory_resource(rt.memory_resource());

    let err = left.with_execution(&ctx).merge(&right, MergeOptions::new()).unwrap_err();

    assert_merge_validation_error(
        &err,
        "dataframe::with_execution::merge",
        "missing_join_keys",
        "set on() or left_on() + right_on()",
    );
}

fn test_merge_reassembly_key_count_invariant_uses_stable_error_contract() {
    let err = cudf::testing::join::trigger_merge_reassembly_key_count_error_for_test().unwrap_err();

    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(err.operation(), "dataframe::merge");
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("join_result_key_count_exceeds_result_columns"));
    assert_eq!(
        error_context_value(&err, "remedy").as_deref(),
        Some(
            "report this as a library bug; join result schema should contain at least one column per key on each side"
        )
    );
    assert_eq!(error_context_value(&err, "keys_count").as_deref(), Some("2"));
    assert_eq!(error_context_value(&err, "left_columns").as_deref(), Some("1"));
    assert_eq!(error_context_value(&err, "right_columns").as_deref(), Some("1"));
    assert_eq!(error_context_value(&err, "num_keys").as_deref(), None);
}

fn test_merge_rejects_ambiguous_join_key_configuration() {
    let left = left_df();
    let right = right_df();

    let err = left.merge(&right, MergeOptions::new().on(["id"]).left_on(["id"]).right_on(["id"])).unwrap_err();

    assert_merge_validation_error(
        &err,
        "dataframe::merge",
        "ambiguous_join_key_configuration",
        "use on() or left_on() + right_on(), not both",
    );
}

fn test_merge_rejects_duplicate_output_names_after_reassembly() {
    let left_ids = column_from_slice(&[1i32, 2, 3]).unwrap();
    let left_conflicting = column_from_slice(&[10i32, 20, 30]).unwrap();
    let left = crate::common::dataframe_from_columns(
        vec!["left_id".to_string(), "right_id".to_string()],
        vec![left_ids, left_conflicting],
    )
    .unwrap();

    let right_ids = column_from_slice(&[2i32, 3, 4]).unwrap();
    let right_values = column_from_slice(&[200i32, 300, 400]).unwrap();
    let right = crate::common::dataframe_from_columns(
        vec!["right_id".to_string(), "value".to_string()],
        vec![right_ids, right_values],
    )
    .unwrap();

    let err = left.merge(&right, MergeOptions::new().left_on(["left_id"]).right_on(["right_id"])).unwrap_err();

    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.operation(), "dataframe::merge");
    assert_duplicate_name_error(&err, "right_id");
}

// ---------------------------------------------------------------------------
// Multi-key merge
// ---------------------------------------------------------------------------

fn test_merge_multi_key_inner() {
    let lk0 = column_from_slice(&[1i32, 1, 2]).unwrap();
    let lk1 = column_from_slice(&[10i32, 20, 10]).unwrap();
    let lp = column_from_slice(&[100i32, 200, 300]).unwrap();
    let left = crate::common::dataframe_from_columns(
        vec!["a".to_string(), "b".to_string(), "val".to_string()],
        vec![lk0, lk1, lp],
    )
    .unwrap();

    let rk0 = column_from_slice(&[1i32, 2, 2]).unwrap();
    let rk1 = column_from_slice(&[10i32, 10, 20]).unwrap();
    let rp = column_from_slice(&[1000i32, 2000, 3000]).unwrap();
    let right = crate::common::dataframe_from_columns(
        vec!["a".to_string(), "b".to_string(), "rval".to_string()],
        vec![rk0, rk1, rp],
    )
    .unwrap();

    let result = left.merge(&right, MergeOptions::new().on(["a", "b"])).unwrap();

    // Matches: (1,10) and (2,10) → 2 rows.
    assert_eq!(result.num_rows(), 2);
    assert_eq!(result.column_names(), vec!["a", "b", "val", "rval"]);

    let mut rows: Vec<_> = i32_column(&result, "a")
        .into_iter()
        .zip(i32_column(&result, "b"))
        .zip(i32_column(&result, "val"))
        .zip(i32_column(&result, "rval"))
        .map(|(((a, b), val), rval)| (a, b, val, rval))
        .collect();
    rows.sort();
    assert_eq!(rows, vec![(1, 10, 100, 1000), (2, 10, 300, 2000)]);
}

// ---------------------------------------------------------------------------
// NullEquality option
// ---------------------------------------------------------------------------

fn test_merge_null_equality_unequal() {
    // With null_equality(Unequal), null keys should NOT match.
    let left_ids = column_from_slice_nullable(&[1i32, 2, 3], &[true, false, true]).unwrap();
    let left_vals = column_from_slice(&[10i32, 20, 30]).unwrap();
    let left =
        crate::common::dataframe_from_columns(vec!["id".to_string(), "val".to_string()], vec![left_ids, left_vals])
            .unwrap();

    let right_ids = column_from_slice_nullable(&[1i32, 2, 3], &[true, false, true]).unwrap();
    let right_vals = column_from_slice(&[100i32, 200, 300]).unwrap();
    let right =
        crate::common::dataframe_from_columns(vec!["id".to_string(), "rval".to_string()], vec![right_ids, right_vals])
            .unwrap();

    use cudf::NullEquality;

    // With Equal: null == null matches, and the joined key remains null.
    let eq_result = left.merge(&right, MergeOptions::new().on(["id"]).null_equality(NullEquality::Equal)).unwrap();
    assert_eq!(eq_result.num_rows(), 3);
    let mut eq_rows: Vec<_> = nullable_i32_column(&eq_result, "id")
        .into_iter()
        .zip(i32_column(&eq_result, "val"))
        .zip(i32_column(&eq_result, "rval"))
        .map(|((id, val), rval)| (id, val, rval))
        .collect();
    eq_rows.sort();
    assert_eq!(eq_rows, vec![(None, 20, 200), (Some(1), 10, 100), (Some(3), 30, 300)]);

    // With Unequal: null != null → only rows with id=1 and id=3 match → 2 rows.
    let ne_result = left.merge(&right, MergeOptions::new().on(["id"]).null_equality(NullEquality::Unequal)).unwrap();
    assert_eq!(ne_result.num_rows(), 2);
    let mut ne_rows: Vec<_> = nullable_i32_column(&ne_result, "id")
        .into_iter()
        .zip(i32_column(&ne_result, "val"))
        .zip(i32_column(&ne_result, "rval"))
        .map(|((id, val), rval)| (id, val, rval))
        .collect();
    ne_rows.sort();
    assert_eq!(ne_rows, vec![(Some(1), 10, 100), (Some(3), 30, 300)]);
}

#[test]
fn dataframe_merge_tests_cases() {
    crate::common::run_named_cases(&[
        (
            "test_dataframe_left_semi_join_materializes_left_rows_without_right_duplicate_growth",
            test_dataframe_left_semi_join_materializes_left_rows_without_right_duplicate_growth,
        ),
        (
            "test_dataframe_left_anti_join_materializes_only_non_matching_left_rows",
            test_dataframe_left_anti_join_materializes_only_non_matching_left_rows,
        ),
        (
            "test_dataframe_join_indices_materialize_selected_payload_columns",
            test_dataframe_join_indices_materialize_selected_payload_columns,
        ),
        (
            "test_dataframe_join_indices_materialization_reservation_matches_output_allocation",
            test_dataframe_join_indices_materialization_reservation_matches_output_allocation,
        ),
        (
            "test_dataframe_join_indices_materialization_reservation_is_zero_for_empty_gather_map",
            test_dataframe_join_indices_materialization_reservation_is_zero_for_empty_gather_map,
        ),
        (
            "test_projected_cross_join_plan_reports_size_and_materializes_both_sides",
            test_projected_cross_join_plan_reports_size_and_materializes_both_sides,
        ),
        (
            "test_projected_cross_join_plan_supports_left_only_projection",
            test_projected_cross_join_plan_supports_left_only_projection,
        ),
        (
            "test_projected_cross_join_plan_supports_right_only_projection",
            test_projected_cross_join_plan_supports_right_only_projection,
        ),
        (
            "test_legacy_projected_cross_join_supports_one_sided_projection",
            test_legacy_projected_cross_join_supports_one_sided_projection,
        ),
        (
            "test_projected_cross_join_plan_reservation_and_materialize_use_execution_override",
            test_projected_cross_join_plan_reservation_and_materialize_use_execution_override,
        ),
        (
            "test_projected_cross_join_one_sided_projection_does_not_order_unselected_side",
            test_projected_cross_join_one_sided_projection_does_not_order_unselected_side,
        ),
        (
            "test_projected_cross_join_plan_rejects_duplicate_output_names_at_creation",
            test_projected_cross_join_plan_rejects_duplicate_output_names_at_creation,
        ),
        (
            "test_projected_cross_join_plan_rejects_zero_output_columns_explicitly",
            test_projected_cross_join_plan_rejects_zero_output_columns_explicitly,
        ),
        (
            "test_projected_cross_join_plan_rejects_row_count_overflow_before_execution",
            test_projected_cross_join_plan_rejects_row_count_overflow_before_execution,
        ),
        (
            "test_projected_cross_join_plan_rejects_input_row_count_above_cudf_limit_even_with_zero_output",
            test_projected_cross_join_plan_rejects_input_row_count_above_cudf_limit_even_with_zero_output,
        ),
        (
            "test_dataframe_join_indices_materialization_reservation_orders_mixed_domains",
            test_dataframe_join_indices_materialization_reservation_orders_mixed_domains,
        ),
        (
            "test_dataframe_inner_join_indices_materialize_projected_sides_with_duplicates",
            test_dataframe_inner_join_indices_materialize_projected_sides_with_duplicates,
        ),
        (
            "test_dataframe_left_join_indices_nullify_unmatched_right_projection",
            test_dataframe_left_join_indices_nullify_unmatched_right_projection,
        ),
        (
            "test_dataframe_full_join_indices_nullify_unmatched_sides",
            test_dataframe_full_join_indices_nullify_unmatched_sides,
        ),
        (
            "test_dataframe_filtered_join_probe_state_materializes_without_second_size_probe",
            test_dataframe_filtered_join_probe_state_materializes_without_second_size_probe,
        ),
        (
            "test_filtered_join_fixed_width_metadata_preflight_matches_built_owner",
            test_filtered_join_fixed_width_metadata_preflight_matches_built_owner,
        ),
        (
            "test_filtered_join_string_metadata_and_probe_preflights_match_built_owners",
            test_filtered_join_string_metadata_and_probe_preflights_match_built_owners,
        ),
        (
            "test_dataframe_hash_join_reuses_build_for_projected_inner_probes_with_duplicates",
            test_dataframe_hash_join_reuses_build_for_projected_inner_probes_with_duplicates,
        ),
        (
            "test_dataframe_mixed_join_size_data_materializes_inner_and_left_without_recomputing_size",
            test_dataframe_mixed_join_size_data_materializes_inner_and_left_without_recomputing_size,
        ),
        (
            "test_dataframe_hash_join_left_and_full_probes_nullify_unmatched_projection",
            test_dataframe_hash_join_left_and_full_probes_nullify_unmatched_projection,
        ),
        ("test_dataframe_left_semi_join_honors_null_equality", test_dataframe_left_semi_join_honors_null_equality),
        (
            "test_dataframe_left_semi_join_indices_with_execution_uses_borrowed_handles",
            test_dataframe_left_semi_join_indices_with_execution_uses_borrowed_handles,
        ),
        (
            "test_dataframe_inner_join_indices_with_execution_uses_borrowed_handles",
            test_dataframe_inner_join_indices_with_execution_uses_borrowed_handles,
        ),
        (
            "test_dataframe_left_semi_join_rejects_missing_join_keys",
            test_dataframe_left_semi_join_rejects_missing_join_keys,
        ),
        (
            "test_dataframe_left_semi_join_rejects_ambiguous_join_key_configuration",
            test_dataframe_left_semi_join_rejects_ambiguous_join_key_configuration,
        ),
        (
            "test_dataframe_left_anti_join_rejects_left_on_without_right_on",
            test_dataframe_left_anti_join_rejects_left_on_without_right_on,
        ),
        (
            "test_dataframe_left_semi_join_indices_rejects_key_count_mismatch",
            test_dataframe_left_semi_join_indices_rejects_key_count_mismatch,
        ),
        (
            "test_dataframe_materialize_join_indices_reanchors_unknown_column",
            test_dataframe_materialize_join_indices_reanchors_unknown_column,
        ),
        (
            "test_dataframe_left_semi_join_empty_result_preserves_schema",
            test_dataframe_left_semi_join_empty_result_preserves_schema,
        ),
        (
            "test_dataframe_left_anti_join_empty_result_preserves_schema",
            test_dataframe_left_anti_join_empty_result_preserves_schema,
        ),
        ("test_merge_options_defaults", test_merge_options_defaults),
        ("test_merge_options_builder_on", test_merge_options_builder_on),
        ("test_merge_options_builder_left_right_on", test_merge_options_builder_left_right_on),
        ("test_merge_options_builder_how", test_merge_options_builder_how),
        ("test_merge_options_builder_suffixes", test_merge_options_builder_suffixes),
        ("test_inner_merge_same_key", test_inner_merge_same_key),
        (
            "test_merge_succeeds_with_mixed_domains_and_uses_left_provenance",
            test_merge_succeeds_with_mixed_domains_and_uses_left_provenance,
        ),
        ("test_left_merge_asymmetric_keys", test_left_merge_asymmetric_keys),
        ("test_merge_suffix_overlapping_payload", test_merge_suffix_overlapping_payload),
        ("test_merge_custom_suffixes", test_merge_custom_suffixes),
        ("test_merge_no_suffix_when_no_conflict", test_merge_no_suffix_when_no_conflict),
        ("test_full_merge", test_full_merge),
        (
            "test_full_merge_succeeds_with_mixed_domains_and_reassembles_on_left_provenance",
            test_full_merge_succeeds_with_mixed_domains_and_reassembles_on_left_provenance,
        ),
        (
            "test_full_merge_in_context_succeeds_on_mixed_domains_and_reassembles_on_context_resources",
            test_full_merge_in_context_succeeds_on_mixed_domains_and_reassembles_on_context_resources,
        ),
        (
            "test_full_merge_asymmetric_keys_preserves_separate_key_columns",
            test_full_merge_asymmetric_keys_preserves_separate_key_columns,
        ),
        ("test_merge_missing_on_key_in_left", test_merge_missing_on_key_in_left),
        ("test_merge_missing_on_key_in_right", test_merge_missing_on_key_in_right),
        ("test_merge_left_on_without_right_on", test_merge_left_on_without_right_on),
        ("test_merge_right_on_without_left_on", test_merge_right_on_without_left_on),
        ("test_merge_left_on_right_on_length_mismatch", test_merge_left_on_right_on_length_mismatch),
        ("test_merge_no_keys_specified", test_merge_no_keys_specified),
        ("test_merge_in_context_no_keys_specified", test_merge_in_context_no_keys_specified),
        (
            "test_merge_reassembly_key_count_invariant_uses_stable_error_contract",
            test_merge_reassembly_key_count_invariant_uses_stable_error_contract,
        ),
        ("test_merge_rejects_ambiguous_join_key_configuration", test_merge_rejects_ambiguous_join_key_configuration),
        (
            "test_merge_rejects_duplicate_output_names_after_reassembly",
            test_merge_rejects_duplicate_output_names_after_reassembly,
        ),
        ("test_merge_multi_key_inner", test_merge_multi_key_inner),
        ("test_merge_null_equality_unequal", test_merge_null_equality_unequal),
    ]);
}
