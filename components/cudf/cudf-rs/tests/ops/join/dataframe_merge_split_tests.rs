//! Tests for `DataFrame::merge_split`.

use cudf::{Column, DataFrame, DataFrameJoinResult, ExecutionContext, MergeOptions, NullEquality, Series};

use crate::common::{
    last_exec_mr_value, last_exec_stream_value, last_sync_stream_value, reset_exec_observation, reset_sync_observation,
    sample_runtime,
};

fn left_df() -> DataFrame {
    let ids = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let scores = Column::from_slice(&[10i32, 20, 30]).unwrap();
    DataFrame::new(vec!["id".to_string(), "score".to_string()], vec![ids, scores]).unwrap()
}

fn right_df() -> DataFrame {
    let ids = Column::from_slice(&[2i32, 3, 4]).unwrap();
    let values = Column::from_slice(&[200i32, 300, 400]).unwrap();
    DataFrame::new(vec!["id".to_string(), "value".to_string()], vec![ids, values]).unwrap()
}

fn left_df_asym() -> DataFrame {
    let ids = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let scores = Column::from_slice(&[10i32, 20, 30]).unwrap();
    DataFrame::new(vec!["left_id".to_string(), "score".to_string()], vec![ids, scores]).unwrap()
}

fn right_df_asym() -> DataFrame {
    let ids = Column::from_slice(&[2i32, 3, 4]).unwrap();
    let values = Column::from_slice(&[200i32, 300, 400]).unwrap();
    DataFrame::new(vec!["right_id".to_string(), "value".to_string()], vec![ids, values]).unwrap()
}

fn i32_column(df: &DataFrame, name: &str) -> Vec<i32> {
    df.series(name).unwrap().to_vec::<i32>().unwrap()
}

fn nullable_i32_column(df: &DataFrame, name: &str) -> Vec<Option<i32>> {
    let series = df.series(name).unwrap();
    let values = series.to_vec::<i32>().unwrap();
    let validity = series.as_column_view().unwrap().validity().unwrap().unwrap_or_else(|| vec![true; values.len()]);

    values.into_iter().zip(validity).map(|(value, valid)| if valid { Some(value) } else { None }).collect()
}

#[test]
fn test_inner_merge_split_on_shared_key() {
    let left = left_df();
    let right = right_df();

    let opts = MergeOptions::new().on(&["id"]).how_inner();
    let DataFrameJoinResult { left: left_result, right: right_result } = left.merge_split(&right, opts).unwrap();

    assert_eq!(left_result.num_rows(), 2);
    assert_eq!(right_result.num_rows(), 2);
    assert_eq!(left_result.column_names(), vec!["id", "score"]);
    assert_eq!(right_result.column_names(), vec!["id", "value"]);

    let mut left_rows: Vec<_> =
        i32_column(&left_result, "id").into_iter().zip(i32_column(&left_result, "score")).collect();
    left_rows.sort();
    assert_eq!(left_rows, vec![(2, 20), (3, 30)]);

    let mut right_rows: Vec<_> =
        i32_column(&right_result, "id").into_iter().zip(i32_column(&right_result, "value")).collect();
    right_rows.sort();
    assert_eq!(right_rows, vec![(2, 200), (3, 300)]);
}

#[test]
fn test_left_merge_split_preserves_all_left_rows() {
    let left = left_df();
    let right = right_df();

    let opts = MergeOptions::new().on(&["id"]).how_left();
    let DataFrameJoinResult { left: left_result, right: right_result } = left.merge_split(&right, opts).unwrap();

    assert_eq!(left_result.num_rows(), 3);
    assert_eq!(right_result.num_rows(), 3);
    assert_eq!(left_result.column_names(), vec!["id", "score"]);
    assert_eq!(right_result.column_names(), vec!["id", "value"]);

    let mut left_rows: Vec<_> =
        i32_column(&left_result, "id").into_iter().zip(i32_column(&left_result, "score")).collect();
    left_rows.sort();
    assert_eq!(left_rows, vec![(1, 10), (2, 20), (3, 30)]);

    let mut right_rows: Vec<_> = nullable_i32_column(&right_result, "id")
        .into_iter()
        .zip(nullable_i32_column(&right_result, "value"))
        .map(|(id, value)| (id, value))
        .collect();
    right_rows.sort();
    assert_eq!(right_rows, vec![(None, None), (Some(2), Some(200)), (Some(3), Some(300))]);
}

#[test]
fn test_full_merge_split_preserves_all_rows() {
    let left = left_df();
    let right = right_df();

    let opts = MergeOptions::new().on(&["id"]).how_full();
    let DataFrameJoinResult { left: left_result, right: right_result } = left.merge_split(&right, opts).unwrap();

    assert_eq!(left_result.num_rows(), 4);
    assert_eq!(right_result.num_rows(), 4);
    assert_eq!(left_result.column_names(), vec!["id", "score"]);
    assert_eq!(right_result.column_names(), vec!["id", "value"]);

    let mut left_rows: Vec<_> = nullable_i32_column(&left_result, "id")
        .into_iter()
        .zip(nullable_i32_column(&left_result, "score"))
        .map(|(id, score)| (id, score))
        .collect();
    left_rows.sort();
    assert_eq!(left_rows, vec![(None, None), (Some(1), Some(10)), (Some(2), Some(20)), (Some(3), Some(30))]);

    let mut right_rows: Vec<_> = nullable_i32_column(&right_result, "id")
        .into_iter()
        .zip(nullable_i32_column(&right_result, "value"))
        .map(|(id, value)| (id, value))
        .collect();
    right_rows.sort();
    assert_eq!(right_rows, vec![(None, None), (Some(2), Some(200)), (Some(3), Some(300)), (Some(4), Some(400))]);
}

#[test]
fn test_merge_split_asymmetric_keys_preserve_each_side_schema() {
    let left = left_df_asym();
    let right = right_df_asym();

    let opts = MergeOptions::new().left_on(&["left_id"]).right_on(&["right_id"]).how_inner();
    let DataFrameJoinResult { left: left_result, right: right_result } = left.merge_split(&right, opts).unwrap();

    assert_eq!(left_result.num_rows(), 2);
    assert_eq!(right_result.num_rows(), 2);
    assert_eq!(left_result.column_names(), vec!["left_id", "score"]);
    assert_eq!(right_result.column_names(), vec!["right_id", "value"]);

    let mut left_rows: Vec<_> =
        i32_column(&left_result, "left_id").into_iter().zip(i32_column(&left_result, "score")).collect();
    left_rows.sort();
    assert_eq!(left_rows, vec![(2, 20), (3, 30)]);

    let mut right_rows: Vec<_> =
        i32_column(&right_result, "right_id").into_iter().zip(i32_column(&right_result, "value")).collect();
    right_rows.sort();
    assert_eq!(right_rows, vec![(2, 200), (3, 300)]);
}

#[test]
fn test_merge_split_respects_null_equality() {
    let left_ids = Column::from_slice_nullable(&[1i32, 2, 3], &[true, false, true]).unwrap();
    let left_vals = Column::from_slice(&[10i32, 20, 30]).unwrap();
    let left = DataFrame::new(vec!["id".to_string(), "val".to_string()], vec![left_ids, left_vals]).unwrap();

    let right_ids = Column::from_slice_nullable(&[1i32, 2, 3], &[true, false, true]).unwrap();
    let right_vals = Column::from_slice(&[100i32, 200, 300]).unwrap();
    let right = DataFrame::new(vec!["id".to_string(), "rval".to_string()], vec![right_ids, right_vals]).unwrap();

    let eq_result =
        left.merge_split(&right, MergeOptions::new().on(&["id"]).null_equality(NullEquality::Equal)).unwrap();
    assert_eq!(eq_result.left.num_rows(), 3);
    assert_eq!(eq_result.right.num_rows(), 3);
    let mut eq_left_rows: Vec<_> =
        nullable_i32_column(&eq_result.left, "id").into_iter().zip(i32_column(&eq_result.left, "val")).collect();
    eq_left_rows.sort();
    assert_eq!(eq_left_rows, vec![(None, 20), (Some(1), 10), (Some(3), 30)]);
    let mut eq_right_rows: Vec<_> =
        nullable_i32_column(&eq_result.right, "id").into_iter().zip(i32_column(&eq_result.right, "rval")).collect();
    eq_right_rows.sort();
    assert_eq!(eq_right_rows, vec![(None, 200), (Some(1), 100), (Some(3), 300)]);

    let ne_result =
        left.merge_split(&right, MergeOptions::new().on(&["id"]).null_equality(NullEquality::Unequal)).unwrap();
    assert_eq!(ne_result.left.num_rows(), 2);
    assert_eq!(ne_result.right.num_rows(), 2);
    let mut ne_left_rows: Vec<_> =
        nullable_i32_column(&ne_result.left, "id").into_iter().zip(i32_column(&ne_result.left, "val")).collect();
    ne_left_rows.sort();
    assert_eq!(ne_left_rows, vec![(Some(1), 10), (Some(3), 30)]);
    let mut ne_right_rows: Vec<_> =
        nullable_i32_column(&ne_result.right, "id").into_iter().zip(i32_column(&ne_result.right, "rval")).collect();
    ne_right_rows.sort();
    assert_eq!(ne_right_rows, vec![(Some(1), 100), (Some(3), 300)]);
}

#[test]
fn test_merge_split_with_context_succeeds_on_mixed_domains_and_uses_context_resources() {
    let left_rt = sample_runtime();
    let right_rt = sample_runtime();
    let target_rt = sample_runtime();

    let left = DataFrame::from_series([
        ("id", Series::new_with_runtime(&left_rt, "id", &[1i32, 2, 3]).unwrap()),
        ("score", Series::new_with_runtime(&left_rt, "score", &[10i32, 20, 30]).unwrap()),
    ])
    .unwrap();
    let right = DataFrame::from_series([
        ("id", Series::new_with_runtime(&right_rt, "id", &[2i32, 3, 4]).unwrap()),
        ("value", Series::new_with_runtime(&right_rt, "value", &[200i32, 300, 400]).unwrap()),
    ])
    .unwrap();
    let ctx = ExecutionContext::new().with_stream(target_rt.stream()).with_memory_resource(target_rt.memory_resource());
    let expected_stream = target_rt.stream().__debug_raw_handle_value();
    let expected_mr = target_rt.memory_resource().__debug_raw_handle_value();

    reset_exec_observation();
    let split = left.merge_split_with_context(&right, MergeOptions::new().how_inner().on(&["id"]), &ctx).unwrap();

    assert_eq!(split.left.__debug_execution_domain_id(), left.__debug_execution_domain_id());
    assert_eq!(split.right.__debug_execution_domain_id(), left.__debug_execution_domain_id());
    assert_eq!(split.left.__debug_execution_domain_stream_value(), expected_stream);
    assert_eq!(split.right.__debug_execution_domain_stream_value(), expected_stream);
    assert_eq!(split.left.__debug_execution_domain_mr_value(), expected_mr);
    assert_eq!(split.right.__debug_execution_domain_mr_value(), expected_mr);
    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_eq!(last_exec_mr_value(), expected_mr);

    reset_sync_observation();
    let mut left_rows: Vec<_> =
        i32_column(&split.left, "id").into_iter().zip(i32_column(&split.left, "score")).collect();
    let mut right_rows: Vec<_> =
        i32_column(&split.right, "id").into_iter().zip(i32_column(&split.right, "value")).collect();
    left_rows.sort();
    right_rows.sort();
    assert_eq!(left_rows, vec![(2, 20), (3, 30)]);
    assert_eq!(right_rows, vec![(2, 200), (3, 300)]);
    assert_eq!(last_sync_stream_value(), expected_stream);
}

#[test]
fn test_merge_split_succeeds_with_mixed_domains_and_uses_left_provenance() {
    let left_rt = sample_runtime();
    let right_rt = sample_runtime();
    let left = DataFrame::from_series([
        ("id", Series::new_with_runtime(&left_rt, "id", &[1i32, 2, 3]).unwrap()),
        ("score", Series::new_with_runtime(&left_rt, "score", &[10i32, 20, 30]).unwrap()),
    ])
    .unwrap();
    let right = DataFrame::from_series([
        ("id", Series::new_with_runtime(&right_rt, "id", &[2i32, 3, 4]).unwrap()),
        ("value", Series::new_with_runtime(&right_rt, "value", &[200i32, 300, 400]).unwrap()),
    ])
    .unwrap();
    let expected_exec_stream = left.__debug_execution_domain_stream_value();
    let expected_exec_mr = left.__debug_execution_domain_mr_value();
    let expected_sync_stream = right.__debug_execution_domain_stream_value();

    reset_exec_observation();
    let split = left.merge_split(&right, MergeOptions::new().how_inner().on(&["id"])).unwrap();

    assert_eq!(split.left.__debug_execution_domain_id(), left.__debug_execution_domain_id());
    assert_eq!(split.right.__debug_execution_domain_id(), left.__debug_execution_domain_id());
    assert_eq!(split.left.__debug_execution_domain_mr_value(), expected_exec_mr);
    assert_eq!(split.right.__debug_execution_domain_mr_value(), expected_exec_mr);
    assert_eq!(last_exec_stream_value(), expected_exec_stream);
    assert_eq!(last_exec_mr_value(), expected_exec_mr);
    assert_eq!(last_sync_stream_value(), expected_sync_stream);

    reset_sync_observation();
    let left_rows: Vec<_> = i32_column(&split.left, "id").into_iter().zip(i32_column(&split.left, "score")).collect();
    let right_rows: Vec<_> =
        i32_column(&split.right, "id").into_iter().zip(i32_column(&split.right, "value")).collect();
    assert_eq!(left_rows, vec![(2, 20), (3, 30)]);
    assert_eq!(right_rows, vec![(2, 200), (3, 300)]);
    assert_eq!(last_sync_stream_value(), expected_exec_stream);
}
