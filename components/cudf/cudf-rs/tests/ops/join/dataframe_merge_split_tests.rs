//! Tests for `DataFrame::merge_split`.
use cudf::{
    DataFrame, ErrorCode, ErrorKind, ExecutionOverrides, MergeOptions, NullEquality,
    frame::dataframe::DataFrameJoinSides, testing::prelude::*,
};

use crate::common::{
    column_from_slice, column_from_slice_nullable, error_context_value, last_exec_mr_value, last_exec_stream_value,
    last_sync_stream_value, reset_exec_observation, reset_sync_observation, sample_runtime,
};

fn left_df() -> DataFrame {
    let ids = column_from_slice(&[1i32, 2, 3]).unwrap();
    let scores = column_from_slice(&[10i32, 20, 30]).unwrap();
    crate::common::dataframe_from_columns(vec!["id".to_string(), "score".to_string()], vec![ids, scores]).unwrap()
}

fn right_df() -> DataFrame {
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

fn i32_column(df: &DataFrame, name: &str) -> Vec<i32> {
    df.series(name).unwrap().to_optional_vec::<i32>().unwrap().into_iter().map(|v| v.unwrap()).collect()
}

fn nullable_i32_column(df: &DataFrame, name: &str) -> Vec<Option<i32>> {
    df.series(name).unwrap().to_optional_vec::<i32>().unwrap()
}

fn assert_merge_split_validation_error(err: &cudf::Error, operation: &str, reason: &str, remedy: &str) {
    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(err.operation(), operation);
    assert_eq!(error_context_value(err, "reason").as_deref(), Some(reason));
    assert_eq!(error_context_value(err, "remedy").as_deref(), Some(remedy));
}

#[test]
fn test_inner_merge_split_on_shared_key() {
    let left = left_df();
    let right = right_df();

    let opts = MergeOptions::new().on(["id"]).how_inner();
    let DataFrameJoinSides { left: left_result, right: right_result } = left.merge_split(&right, opts).unwrap();

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

    let opts = MergeOptions::new().on(["id"]).how_left();
    let DataFrameJoinSides { left: left_result, right: right_result } = left.merge_split(&right, opts).unwrap();

    assert_eq!(left_result.num_rows(), 3);
    assert_eq!(right_result.num_rows(), 3);
    assert_eq!(left_result.column_names(), vec!["id", "score"]);
    assert_eq!(right_result.column_names(), vec!["id", "value"]);

    let mut left_rows: Vec<_> =
        i32_column(&left_result, "id").into_iter().zip(i32_column(&left_result, "score")).collect();
    left_rows.sort();
    assert_eq!(left_rows, vec![(1, 10), (2, 20), (3, 30)]);

    let mut right_rows: Vec<_> =
        nullable_i32_column(&right_result, "id").into_iter().zip(nullable_i32_column(&right_result, "value")).collect();
    right_rows.sort();
    assert_eq!(right_rows, vec![(None, None), (Some(2), Some(200)), (Some(3), Some(300))]);
}

#[test]
fn test_full_merge_split_preserves_all_rows() {
    let left = left_df();
    let right = right_df();

    let opts = MergeOptions::new().on(["id"]).how_full();
    let DataFrameJoinSides { left: left_result, right: right_result } = left.merge_split(&right, opts).unwrap();

    assert_eq!(left_result.num_rows(), 4);
    assert_eq!(right_result.num_rows(), 4);
    assert_eq!(left_result.column_names(), vec!["id", "score"]);
    assert_eq!(right_result.column_names(), vec!["id", "value"]);

    let mut left_rows: Vec<_> =
        nullable_i32_column(&left_result, "id").into_iter().zip(nullable_i32_column(&left_result, "score")).collect();
    left_rows.sort();
    assert_eq!(left_rows, vec![(None, None), (Some(1), Some(10)), (Some(2), Some(20)), (Some(3), Some(30))]);

    let mut right_rows: Vec<_> =
        nullable_i32_column(&right_result, "id").into_iter().zip(nullable_i32_column(&right_result, "value")).collect();
    right_rows.sort();
    assert_eq!(right_rows, vec![(None, None), (Some(2), Some(200)), (Some(3), Some(300)), (Some(4), Some(400))]);
}

#[test]
fn test_merge_split_asymmetric_keys_preserve_each_side_schema() {
    let left = left_df_asym();
    let right = right_df_asym();

    let opts = MergeOptions::new().left_on(["left_id"]).right_on(["right_id"]).how_inner();
    let DataFrameJoinSides { left: left_result, right: right_result } = left.merge_split(&right, opts).unwrap();

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

    let eq_result =
        left.merge_split(&right, MergeOptions::new().on(["id"]).null_equality(NullEquality::Equal)).unwrap();
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
        left.merge_split(&right, MergeOptions::new().on(["id"]).null_equality(NullEquality::Unequal)).unwrap();
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
fn test_merge_split_in_context_succeeds_on_mixed_domains_and_uses_context_resources() {
    let left_rt = sample_runtime();
    let right_rt = sample_runtime();
    let target_rt = sample_runtime();

    let left = DataFrame::from_columns([
        ("id", left_rt.series(&[1i32, 2, 3]).unwrap()),
        ("score", left_rt.series(&[10i32, 20, 30]).unwrap()),
    ])
    .unwrap();
    let right = DataFrame::from_columns([
        ("id", right_rt.series(&[2i32, 3, 4]).unwrap()),
        ("value", right_rt.series(&[200i32, 300, 400]).unwrap()),
    ])
    .unwrap();
    let ctx =
        ExecutionOverrides::new().with_stream(target_rt.stream()).with_memory_resource(target_rt.memory_resource());
    let expected_stream = target_rt.stream().testing_raw_handle_value();
    let expected_mr = target_rt.memory_resource().testing_raw_handle_value();

    reset_exec_observation();
    let split = left.with_execution(&ctx).merge_split(&right, MergeOptions::new().how_inner().on(["id"])).unwrap();

    assert_eq!(split.left.testing_execution_domain_id(), left.testing_execution_domain_id());
    assert_eq!(split.right.testing_execution_domain_id(), left.testing_execution_domain_id());
    assert_eq!(split.left.testing_execution_domain_stream_value(), expected_stream);
    assert_eq!(split.right.testing_execution_domain_stream_value(), expected_stream);
    assert_eq!(split.left.testing_execution_domain_mr_value(), expected_mr);
    assert_eq!(split.right.testing_execution_domain_mr_value(), expected_mr);
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
    let _event_observation = crate::common::lock_event_observation();
    let left_rt = sample_runtime();
    let right_rt = sample_runtime();
    let left = DataFrame::from_columns([
        ("id", left_rt.series(&[1i32, 2, 3]).unwrap()),
        ("score", left_rt.series(&[10i32, 20, 30]).unwrap()),
    ])
    .unwrap();
    let right = DataFrame::from_columns([
        ("id", right_rt.series(&[2i32, 3, 4]).unwrap()),
        ("value", right_rt.series(&[200i32, 300, 400]).unwrap()),
    ])
    .unwrap();
    let expected_exec_stream = left.testing_execution_domain_stream_value();
    let expected_exec_mr = left.testing_execution_domain_mr_value();
    crate::common::reset_dependency_observation();
    reset_exec_observation();
    let split = left.merge_split(&right, MergeOptions::new().how_inner().on(["id"])).unwrap();

    assert_eq!(split.left.testing_execution_domain_id(), left.testing_execution_domain_id());
    assert_eq!(split.right.testing_execution_domain_id(), left.testing_execution_domain_id());
    assert_eq!(split.left.testing_execution_domain_mr_value(), expected_exec_mr);
    assert_eq!(split.right.testing_execution_domain_mr_value(), expected_exec_mr);
    assert_eq!(last_exec_stream_value(), expected_exec_stream);
    assert_eq!(last_exec_mr_value(), expected_exec_mr);
    // DataFrame guarantees one retained provenance bundle for all columns, so
    // one dependency from the right frame's stream orders both right keys and
    // payload before the left/output stream.
    assert_eq!(crate::common::dependency_wait_call_count_for_stream(expected_exec_stream), 1);

    reset_sync_observation();
    let left_rows: Vec<_> = i32_column(&split.left, "id").into_iter().zip(i32_column(&split.left, "score")).collect();
    let right_rows: Vec<_> =
        i32_column(&split.right, "id").into_iter().zip(i32_column(&split.right, "value")).collect();
    assert_eq!(left_rows, vec![(2, 20), (3, 30)]);
    assert_eq!(right_rows, vec![(2, 200), (3, 300)]);
    assert_eq!(last_sync_stream_value(), expected_exec_stream);
}

#[test]
fn test_merge_split_left_on_without_right_on() {
    let left = left_df();
    let right = right_df();

    let err = left.merge_split(&right, MergeOptions::new().left_on(["id"])).unwrap_err();

    assert_merge_split_validation_error(
        &err,
        "dataframe::merge_split",
        "left_on_without_right_on",
        "set right_on() or use on()",
    );
}

#[test]
fn test_merge_split_right_on_without_left_on() {
    let left = left_df();
    let right = right_df();

    let err = left.merge_split(&right, MergeOptions::new().right_on(["id"])).unwrap_err();

    assert_merge_split_validation_error(
        &err,
        "dataframe::merge_split",
        "right_on_without_left_on",
        "set left_on() or use on()",
    );
}

#[test]
fn test_merge_split_left_on_right_on_length_mismatch() {
    let left = left_df_asym();
    let right = right_df_asym();

    let err =
        left.merge_split(&right, MergeOptions::new().left_on(["left_id"]).right_on(["right_id", "value"])).unwrap_err();

    assert_merge_split_validation_error(
        &err,
        "dataframe::merge_split",
        "join_key_count_mismatch",
        "set left_on() and right_on() with the same number of key columns",
    );
    assert_eq!(error_context_value(&err, "left_on_count").as_deref(), Some("1"));
    assert_eq!(error_context_value(&err, "right_on_count").as_deref(), Some("2"));
}

#[test]
fn test_merge_split_no_keys_specified() {
    let left = left_df();
    let right = right_df();

    let err = left.merge_split(&right, MergeOptions::new()).unwrap_err();

    assert_merge_split_validation_error(
        &err,
        "dataframe::merge_split",
        "missing_join_keys",
        "set on() or left_on() + right_on()",
    );
}

#[test]
fn test_merge_split_rejects_ambiguous_join_key_configuration() {
    let left = left_df();
    let right = right_df();

    let err = left.merge_split(&right, MergeOptions::new().on(["id"]).left_on(["id"]).right_on(["id"])).unwrap_err();

    assert_merge_split_validation_error(
        &err,
        "dataframe::merge_split",
        "ambiguous_join_key_configuration",
        "use on() or left_on() + right_on(), not both",
    );
}

#[test]
fn test_merge_split_in_context_left_on_without_right_on() {
    let left = left_df();
    let right = right_df();
    let rt = sample_runtime();
    let ctx = ExecutionOverrides::new().with_stream(rt.stream()).with_memory_resource(rt.memory_resource());

    let err = left.with_execution(&ctx).merge_split(&right, MergeOptions::new().left_on(["id"])).unwrap_err();

    assert_merge_split_validation_error(
        &err,
        "dataframe::with_execution::merge_split",
        "left_on_without_right_on",
        "set right_on() or use on()",
    );
}
