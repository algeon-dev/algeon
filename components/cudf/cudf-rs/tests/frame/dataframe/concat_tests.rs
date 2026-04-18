//! Tests for DataFrame concat, copy, rename, select.
use cudf::testing::prelude::*;


use cudf::{ColumnOrder, ConcatByNameOptions, DataFrame, ErrorKind, SchemaAlignment};

use crate::common::{
    assert_missing_column_error, column_from_slice, last_exec_mr_value, last_exec_stream_value, last_sync_stream_value,
    reset_exec_observation, reset_sync_observation, sample_runtime,
};

#[test]
fn test_concat_matching_schemas() {
    let df1 = crate::common::dataframe_from_columns(
        vec!["a".to_string(), "b".to_string()],
        vec![column_from_slice(&[1i32, 2]).unwrap(), column_from_slice(&[10.0f64, 20.0]).unwrap()],
    )
    .unwrap();

    let df2 = crate::common::dataframe_from_columns(
        vec!["a".to_string(), "b".to_string()],
        vec![column_from_slice(&[3i32, 4]).unwrap(), column_from_slice(&[30.0f64, 40.0]).unwrap()],
    )
    .unwrap();

    let result = df1.concat(&df2).unwrap();

    assert_eq!(result.num_columns(), 2);
    assert_eq!(result.num_rows(), 4);
    assert_eq!(result.column_names(), vec!["a", "b"]);

    let a_vals: Vec<i32> = result.series("a").unwrap().to_vec_strict::<i32>().unwrap();
    assert_eq!(a_vals, vec![1, 2, 3, 4]);

    let b_vals: Vec<f64> = result.series("b").unwrap().to_vec_strict::<f64>().unwrap();
    assert_eq!(b_vals, vec![10.0, 20.0, 30.0, 40.0]);
}

#[test]
fn test_concat_mismatched_column_names_returns_error() {
    let df1 = crate::common::dataframe_from_columns(
        vec!["a".to_string(), "b".to_string()],
        vec![column_from_slice(&[1i32, 2]).unwrap(), column_from_slice(&[10i32, 20]).unwrap()],
    )
    .unwrap();

    let df2 = crate::common::dataframe_from_columns(
        vec!["a".to_string(), "c".to_string()],
        vec![column_from_slice(&[3i32, 4]).unwrap(), column_from_slice(&[30i32, 40]).unwrap()],
    )
    .unwrap();

    let err = df1.concat(&df2).expect_err("expected schema mismatch error");
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
}

#[test]
fn test_concat_mismatched_column_count_returns_error() {
    let df1 =
        crate::common::dataframe_from_columns(vec!["a".to_string()], vec![column_from_slice(&[1i32, 2]).unwrap()])
            .unwrap();

    let df2 = crate::common::dataframe_from_columns(
        vec!["a".to_string(), "b".to_string()],
        vec![column_from_slice(&[3i32, 4]).unwrap(), column_from_slice(&[30i32, 40]).unwrap()],
    )
    .unwrap();

    let err = df1.concat(&df2).expect_err("expected schema mismatch error");
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
}

#[test]
fn test_concat_mismatched_column_types_returns_error() {
    let df1 = crate::common::dataframe_from_columns(
        vec!["a".to_string(), "b".to_string()],
        vec![column_from_slice(&[1i32, 2]).unwrap(), column_from_slice(&[10i32, 20]).unwrap()],
    )
    .unwrap();

    // Same schema names/order, but "b" has different type (i64 vs i32).
    let df2 = crate::common::dataframe_from_columns(
        vec!["a".to_string(), "b".to_string()],
        vec![column_from_slice(&[3i32, 4]).unwrap(), column_from_slice(&[30i64, 40]).unwrap()],
    )
    .unwrap();

    let err = df1.concat(&df2).expect_err("expected type mismatch error");
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
}

#[test]
fn test_concat_still_requires_exact_schema_even_when_concat_by_name_can_reorder() {
    let df1 = crate::common::dataframe_from_columns(
        vec!["a".to_string(), "b".to_string()],
        vec![column_from_slice(&[1i32, 2]).unwrap(), column_from_slice(&[10i32, 20]).unwrap()],
    )
    .unwrap();

    let df2 = crate::common::dataframe_from_columns(
        vec!["b".to_string(), "a".to_string()],
        vec![column_from_slice(&[30i32, 40]).unwrap(), column_from_slice(&[3i32, 4]).unwrap()],
    )
    .unwrap();

    assert!(df1.concat(&df2).is_err());

    let result = df1.concat_by_name(&df2, ConcatByNameOptions::new().alignment(SchemaAlignment::Exact)).unwrap();

    assert_eq!(result.column_names(), vec!["a", "b"]);
    assert_eq!(result.series("a").unwrap().to_vec_strict::<i32>().unwrap(), vec![1, 2, 3, 4]);
    assert_eq!(result.series("b").unwrap().to_vec_strict::<i32>().unwrap(), vec![10, 20, 30, 40]);
}

#[test]
fn test_concat_by_name_null_fill_aligns_one_sided_missing_column() {
    let left =
        crate::common::dataframe_from_columns(vec!["id".to_string()], vec![column_from_slice(&[1i32, 2]).unwrap()])
            .unwrap();

    let right = crate::common::dataframe_from_columns(
        vec!["id".to_string(), "score".to_string()],
        vec![column_from_slice(&[3i32, 4]).unwrap(), column_from_slice(&[30.0f64, 40.0]).unwrap()],
    )
    .unwrap();

    let result = left.concat_by_name(&right, ConcatByNameOptions::default()).unwrap();

    assert_eq!(result.column_names(), vec!["id", "score"]);
    assert_eq!(result.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![1, 2, 3, 4]);
    assert_eq!(
        result.series("score").unwrap().to_optional_vec::<f64>().unwrap(),
        vec![None, None, Some(30.0), Some(40.0)]
    );
}

#[test]
fn test_concat_many_by_name_handles_multi_input_union() {
    let left = crate::common::dataframe_from_columns(vec!["id".to_string()], vec![column_from_slice(&[1i32]).unwrap()])
        .unwrap();
    let middle = crate::common::dataframe_from_columns(
        vec!["city".to_string()],
        vec![crate::common::column_from_strs(&["Toronto"]).unwrap()],
    )
    .unwrap();
    let right =
        crate::common::dataframe_from_columns(vec!["score".to_string()], vec![column_from_slice(&[99.0f64]).unwrap()])
            .unwrap();

    let result = left.concat_many_by_name(&[&middle, &right], ConcatByNameOptions::default()).unwrap();

    assert_eq!(result.column_names(), vec!["id", "city", "score"]);
    assert_eq!(result.series("id").unwrap().to_optional_vec::<i32>().unwrap(), vec![Some(1), None, None]);
    assert_eq!(
        result.series("city").unwrap().to_optional_strings().unwrap(),
        vec![None, Some("Toronto".to_string()), None]
    );
    assert_eq!(result.series("score").unwrap().to_optional_vec::<f64>().unwrap(), vec![None, None, Some(99.0)]);
}

#[test]
fn test_concat_by_name_honors_left_column_order() {
    let left = crate::common::dataframe_from_columns(
        vec!["b".to_string(), "a".to_string()],
        vec![column_from_slice(&[1i32]).unwrap(), column_from_slice(&[2i32]).unwrap()],
    )
    .unwrap();
    let right = crate::common::dataframe_from_columns(
        vec!["c".to_string(), "a".to_string()],
        vec![column_from_slice(&[3i32]).unwrap(), column_from_slice(&[4i32]).unwrap()],
    )
    .unwrap();

    let result = left
        .concat_by_name(
            &right,
            ConcatByNameOptions::new().alignment(SchemaAlignment::NullFill).column_order(ColumnOrder::Left),
        )
        .unwrap();

    assert_eq!(result.column_names(), vec!["b", "a", "c"]);
}

#[test]
fn test_concat_by_name_honors_first_seen_column_order() {
    let left = crate::common::dataframe_from_columns(vec!["id".to_string()], vec![column_from_slice(&[1i32]).unwrap()])
        .unwrap();
    let middle = crate::common::dataframe_from_columns(
        vec!["city".to_string(), "score".to_string()],
        vec![crate::common::column_from_strs(&["Toronto"]).unwrap(), column_from_slice(&[10.0f64]).unwrap()],
    )
    .unwrap();
    let right = crate::common::dataframe_from_columns(
        vec!["score".to_string(), "country".to_string()],
        vec![column_from_slice(&[20.0f64]).unwrap(), crate::common::column_from_strs(&["CA"]).unwrap()],
    )
    .unwrap();

    let result = left
        .concat_many_by_name(
            &[&middle, &right],
            ConcatByNameOptions::new().alignment(SchemaAlignment::NullFill).column_order(ColumnOrder::FirstSeen),
        )
        .unwrap();

    assert_eq!(result.column_names(), vec!["id", "city", "score", "country"]);
}

#[test]
fn test_concat_by_name_non_null_dtype_mismatch_still_errors_under_null_fill() {
    let left =
        crate::common::dataframe_from_columns(vec!["id".to_string()], vec![column_from_slice(&[1i32, 2]).unwrap()])
            .unwrap();
    let right =
        crate::common::dataframe_from_columns(vec!["id".to_string()], vec![column_from_slice(&[3i64, 4]).unwrap()])
            .unwrap();

    let err = left.concat_by_name(&right, ConcatByNameOptions::default()).unwrap_err();

    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
}

#[test]
fn test_concat_by_name_exact_rejects_missing_columns() {
    let left = crate::common::dataframe_from_columns(
        vec!["id".to_string(), "score".to_string()],
        vec![column_from_slice(&[1i32]).unwrap(), column_from_slice(&[10.0f64]).unwrap()],
    )
    .unwrap();
    let right =
        crate::common::dataframe_from_columns(vec!["id".to_string()], vec![column_from_slice(&[2i32]).unwrap()])
            .unwrap();

    let err = left.concat_by_name(&right, ConcatByNameOptions::new().alignment(SchemaAlignment::Exact)).unwrap_err();

    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
}

#[test]
fn test_concat_succeeds_with_mixed_domains_and_uses_left_provenance() {
    let left_rt = sample_runtime();
    let right_rt = sample_runtime();
    let left = DataFrame::from_series([left_rt.series("a", &[1i32, 2]).unwrap()]).unwrap();
    let right = DataFrame::from_series([right_rt.series("a", &[3i32, 4]).unwrap()]).unwrap();
    let expected_exec_stream = left.testing_execution_domain_stream_value();
    let expected_exec_mr = left.testing_execution_domain_mr_value();
    let expected_sync_stream = right.testing_execution_domain_stream_value();

    reset_exec_observation();
    let result = left.concat(&right).unwrap();

    assert_eq!(result.testing_execution_domain_id(), left.testing_execution_domain_id());
    assert_eq!(result.testing_execution_domain_mr_value(), expected_exec_mr);
    assert_eq!(last_exec_stream_value(), expected_exec_stream);
    assert_eq!(last_exec_mr_value(), expected_exec_mr);
    assert_eq!(last_sync_stream_value(), expected_sync_stream);

    reset_sync_observation();
    let host = result.series("a").unwrap().to_vec_strict::<i32>().unwrap();
    assert_eq!(host, vec![1, 2, 3, 4]);
    assert_eq!(last_sync_stream_value(), expected_exec_stream);
}

#[test]
fn test_concat_many_three_dataframes() {
    let df1 = crate::common::dataframe_from_columns(
        vec!["a".to_string(), "b".to_string()],
        vec![column_from_slice(&[1i32, 2]).unwrap(), column_from_slice(&[10.0f64, 20.0]).unwrap()],
    )
    .unwrap();

    let df2 = crate::common::dataframe_from_columns(
        vec!["a".to_string(), "b".to_string()],
        vec![column_from_slice(&[3i32, 4]).unwrap(), column_from_slice(&[30.0f64, 40.0]).unwrap()],
    )
    .unwrap();

    let df3 = crate::common::dataframe_from_columns(
        vec!["a".to_string(), "b".to_string()],
        vec![column_from_slice(&[5i32, 6]).unwrap(), column_from_slice(&[50.0f64, 60.0]).unwrap()],
    )
    .unwrap();

    let result = df1.concat_many(&[&df2, &df3]).unwrap();

    assert_eq!(result.num_columns(), 2);
    assert_eq!(result.num_rows(), 6);
    assert_eq!(result.column_names(), vec!["a", "b"]);

    let a_vals: Vec<i32> = result.series("a").unwrap().to_vec_strict::<i32>().unwrap();
    assert_eq!(a_vals, vec![1, 2, 3, 4, 5, 6]);

    let b_vals: Vec<f64> = result.series("b").unwrap().to_vec_strict::<f64>().unwrap();
    assert_eq!(b_vals, vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
}

#[test]
fn test_concat_many_empty_slice_returns_self() {
    let df =
        crate::common::dataframe_from_columns(vec!["a".to_string()], vec![column_from_slice(&[1i32, 2, 3]).unwrap()])
            .unwrap();

    let result = df.concat_many(&[]).unwrap();

    assert_eq!(result.num_columns(), 1);
    assert_eq!(result.num_rows(), 3);
    let a_vals: Vec<i32> = result.series("a").unwrap().to_vec_strict::<i32>().unwrap();
    assert_eq!(a_vals, vec![1, 2, 3]);
}

#[test]
fn test_concat_many_single_other() {
    let df1 =
        crate::common::dataframe_from_columns(vec!["x".to_string()], vec![column_from_slice(&[1i32, 2]).unwrap()])
            .unwrap();

    let df2 =
        crate::common::dataframe_from_columns(vec!["x".to_string()], vec![column_from_slice(&[3i32, 4]).unwrap()])
            .unwrap();

    let result = df1.concat_many(&[&df2]).unwrap();

    assert_eq!(result.num_rows(), 4);
    let vals: Vec<i32> = result.series("x").unwrap().to_vec_strict::<i32>().unwrap();
    assert_eq!(vals, vec![1, 2, 3, 4]);
}

#[test]
fn test_concat_many_mismatched_schema_returns_error() {
    let df1 =
        crate::common::dataframe_from_columns(vec!["a".to_string()], vec![column_from_slice(&[1i32, 2]).unwrap()])
            .unwrap();

    let df2 =
        crate::common::dataframe_from_columns(vec!["a".to_string()], vec![column_from_slice(&[3i32, 4]).unwrap()])
            .unwrap();

    let df3 =
        crate::common::dataframe_from_columns(vec!["b".to_string()], vec![column_from_slice(&[5i32, 6]).unwrap()])
            .unwrap();

    let err = df1.concat_many(&[&df2, &df3]).expect_err("expected schema mismatch");
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
}

#[test]
fn test_concat_many_mismatched_types_returns_error() {
    let df1 =
        crate::common::dataframe_from_columns(vec!["a".to_string()], vec![column_from_slice(&[1i32, 2]).unwrap()])
            .unwrap();

    let df2 =
        crate::common::dataframe_from_columns(vec!["a".to_string()], vec![column_from_slice(&[3i64, 4]).unwrap()])
            .unwrap();

    let err = df1.concat_many(&[&df2]).expect_err("expected type mismatch");
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
}

#[test]
fn test_concat_many_succeeds_with_mixed_domains_and_uses_self_provenance() {
    let rt1 = sample_runtime();
    let rt2 = sample_runtime();
    let rt3 = sample_runtime();
    let df1 = DataFrame::from_series([rt1.series("a", &[1i32, 2]).unwrap()]).unwrap();
    let df2 = DataFrame::from_series([rt2.series("a", &[3i32, 4]).unwrap()]).unwrap();
    let df3 = DataFrame::from_series([rt3.series("a", &[5i32, 6]).unwrap()]).unwrap();
    let expected_domain_id = df1.testing_execution_domain_id();

    let result = df1.concat_many(&[&df2, &df3]).unwrap();

    assert_eq!(result.testing_execution_domain_id(), expected_domain_id);
    let vals: Vec<i32> = result.series("a").unwrap().to_vec_strict::<i32>().unwrap();
    assert_eq!(vals, vec![1, 2, 3, 4, 5, 6]);
}

#[test]
fn test_copy_preserves_schema_and_data() {
    let df = crate::common::dataframe_from_columns(
        vec!["x".to_string(), "y".to_string()],
        vec![column_from_slice(&[1i32, 2, 3]).unwrap(), column_from_slice(&[4.0f64, 5.0, 6.0]).unwrap()],
    )
    .unwrap();

    let copy = df.clone();

    assert_eq!(copy.num_columns(), 2);
    assert_eq!(copy.num_rows(), 3);
    assert_eq!(copy.column_names(), vec!["x", "y"]);

    let x_vals: Vec<i32> = copy.series("x").unwrap().to_vec_strict::<i32>().unwrap();
    assert_eq!(x_vals, vec![1, 2, 3]);

    let y_vals: Vec<f64> = copy.series("y").unwrap().to_vec_strict::<f64>().unwrap();
    assert_eq!(y_vals, vec![4.0, 5.0, 6.0]);
}

#[test]
fn test_rename_column() {
    let df = crate::common::dataframe_from_columns(
        vec!["old_name".to_string(), "other".to_string()],
        vec![column_from_slice(&[1i32, 2]).unwrap(), column_from_slice(&[10i32, 20]).unwrap()],
    )
    .unwrap();

    let renamed = df.rename_column("old_name", "new_name").unwrap();

    // New name works.
    assert_eq!(renamed.column_names(), vec!["new_name", "other"]);
    let vals: Vec<i32> = renamed.series("new_name").unwrap().to_vec_strict::<i32>().unwrap();
    assert_eq!(vals, vec![1, 2]);

    // Old name no longer exists.
    assert!(renamed.series("old_name").is_err());

    // Other column is preserved.
    let other_vals: Vec<i32> = renamed.series("other").unwrap().to_vec_strict::<i32>().unwrap();
    assert_eq!(other_vals, vec![10, 20]);
}

#[test]
fn test_rename_nonexistent_column_returns_error() {
    let df = crate::common::dataframe_from_columns(vec!["a".to_string()], vec![column_from_slice(&[1i32]).unwrap()])
        .unwrap();

    let err = df.rename_column("nonexistent", "new").expect_err("expected error");
    assert_missing_column_error(&err, "nonexistent");
}

#[test]
fn test_rename_to_existing_name_returns_error() {
    let df = crate::common::dataframe_from_columns(
        vec!["a".to_string(), "b".to_string()],
        vec![column_from_slice(&[1i32]).unwrap(), column_from_slice(&[2i32]).unwrap()],
    )
    .unwrap();

    let err = df.rename_column("a", "b").expect_err("expected error");
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
}

#[test]
fn test_select_subset_of_columns() {
    let df = crate::common::dataframe_from_columns(
        vec!["a".to_string(), "b".to_string(), "c".to_string()],
        vec![
            column_from_slice(&[1i32, 2]).unwrap(),
            column_from_slice(&[10.0f64, 20.0]).unwrap(),
            column_from_slice(&[100i64, 200]).unwrap(),
        ],
    )
    .unwrap();

    let selected = df.select(["c", "a"]).unwrap();

    assert_eq!(selected.num_columns(), 2);
    assert_eq!(selected.num_rows(), 2);
    assert_eq!(selected.column_names(), vec!["c", "a"]);

    let c_vals: Vec<i64> = selected.series("c").unwrap().to_vec_strict::<i64>().unwrap();
    assert_eq!(c_vals, vec![100, 200]);

    let a_vals: Vec<i32> = selected.series("a").unwrap().to_vec_strict::<i32>().unwrap();
    assert_eq!(a_vals, vec![1, 2]);
}

#[test]
fn test_select_nonexistent_column_returns_error() {
    let df = crate::common::dataframe_from_columns(vec!["a".to_string()], vec![column_from_slice(&[1i32, 2]).unwrap()])
        .unwrap();

    let err = df.select(["nonexistent"]).expect_err("expected error");
    assert_missing_column_error(&err, "nonexistent");
}

#[test]
fn test_select_duplicate_column_names_returns_error() {
    let df = crate::common::dataframe_from_columns(
        vec!["a".to_string(), "b".to_string()],
        vec![column_from_slice(&[1i32, 2]).unwrap(), column_from_slice(&[10i32, 20]).unwrap()],
    )
    .unwrap();

    let err = df.select(["a", "a"]).expect_err("expected duplicate selection error");
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
}

#[test]
fn test_select_preserves_data() {
    let df = crate::common::dataframe_from_columns(
        vec!["x".to_string(), "y".to_string()],
        vec![column_from_slice(&[1i32, 2, 3]).unwrap(), column_from_slice(&[4.0f64, 5.0, 6.0]).unwrap()],
    )
    .unwrap();

    // Select single column.
    let selected = df.select(["y"]).unwrap();

    assert_eq!(selected.num_columns(), 1);
    assert_eq!(selected.column_names(), vec!["y"]);

    let y_vals: Vec<f64> = selected.series("y").unwrap().to_vec_strict::<f64>().unwrap();
    assert_eq!(y_vals, vec![4.0, 5.0, 6.0]);
}

#[test]
fn test_select_accepts_string_vec() {
    let df = crate::common::dataframe_from_columns(
        vec!["a".to_string(), "b".to_string()],
        vec![column_from_slice(&[1i32, 2]).unwrap(), column_from_slice(&[10i32, 20]).unwrap()],
    )
    .unwrap();

    let names = vec!["b".to_string(), "a".to_string()];
    let selected = df.select(names).unwrap();

    assert_eq!(selected.column_names(), vec!["b", "a"]);
}

#[test]
fn test_rename_preserves_column_order() {
    let df = crate::common::dataframe_from_columns(
        vec!["a".to_string(), "b".to_string(), "c".to_string()],
        vec![
            column_from_slice(&[1i32]).unwrap(),
            column_from_slice(&[2i32]).unwrap(),
            column_from_slice(&[3i32]).unwrap(),
        ],
    )
    .unwrap();

    let renamed = df.rename_column("b", "beta").unwrap();
    assert_eq!(renamed.column_names(), vec!["a", "beta", "c"]);
}

#[test]
fn test_rename_same_name_is_schema_only_and_preserves_data() {
    let df = crate::common::dataframe_from_columns(
        vec!["a".to_string(), "b".to_string()],
        vec![column_from_slice(&[1i32, 2, 3]).unwrap(), column_from_slice(&[10i32, 20, 30]).unwrap()],
    )
    .unwrap();

    let renamed = df.rename_column("a", "a").unwrap();

    // No schema change.
    assert_eq!(renamed.column_names(), vec!["a", "b"]);

    // Data unchanged.
    assert_eq!(renamed.series("a").unwrap().to_vec_strict::<i32>().unwrap(), vec![1, 2, 3]);
    assert_eq!(renamed.series("b").unwrap().to_vec_strict::<i32>().unwrap(), vec![10, 20, 30]);
}

#[test]
fn test_select_all_columns_returns_full_copy() {
    let df = crate::common::dataframe_from_columns(
        vec!["a".to_string(), "b".to_string()],
        vec![column_from_slice(&[1i32, 2]).unwrap(), column_from_slice(&[3i32, 4]).unwrap()],
    )
    .unwrap();

    let selected = df.select(["a", "b"]).unwrap();
    assert_eq!(selected.num_columns(), 2);
    assert_eq!(selected.num_rows(), 2);
    assert_eq!(selected.column_names(), vec!["a", "b"]);

    let a_vals: Vec<i32> = selected.series("a").unwrap().to_vec_strict::<i32>().unwrap();
    assert_eq!(a_vals, vec![1, 2]);
}
