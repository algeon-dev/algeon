//! Tests for representative high-level DataFrame entrypoint contracts.
//!
//! These tests protect user-visible behavior for the high-level APIs that take
//! `Series` or plain Rust values, while keeping one low-level smoke path
//! through `table_view()` plus `cudf::ops::*`.

use cudf::{
    DataFrame, Series,
    ops::{copying, filter as low_level_filter, search},
};

use crate::common::{
    assert_missing_column_error, column_from_slice, last_exec_mr_value, last_exec_stream_value, last_sync_stream_value,
    reset_exec_observation, reset_sync_observation, sample_runtime, scalar,
};

// ===========================================================================
// Helpers
// ===========================================================================

/// Build a 5-row DataFrame with id (i32) and score (f64) columns.
fn sample_df() -> DataFrame {
    let ids = crate::common::series::<i32>("id", &[10, 20, 30, 40, 50]).unwrap();
    let scores = crate::common::series::<f64>("score", &[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    DataFrame::from_series([ids, scores]).unwrap()
}

// ===========================================================================
// 1. gather(&Series)
// ===========================================================================

#[test]
fn test_gather_with_series() {
    let df = sample_df(); // id: 10,20,30,40,50
    let indices = crate::common::series::<i32>("idx", &[0, 2, 4]).unwrap();
    let result = df.gather(&indices).unwrap();
    assert_eq!(result.num_rows(), 3);
    // Gather produces nullable output; use to_optional_vec.
    assert_eq!(result.series("id").unwrap().to_optional_vec::<i32>().unwrap(), vec![Some(10), Some(30), Some(50)]);
    assert_eq!(
        result.series("score").unwrap().to_optional_vec::<f64>().unwrap(),
        vec![Some(1.0), Some(3.0), Some(5.0)]
    );
}

#[test]
fn test_gather_with_series_succeeds_with_mixed_domains_and_uses_dataframe_provenance() {
    let df_rt = sample_runtime();
    let indices_rt = sample_runtime();
    let df = DataFrame::from_series([
        df_rt.series("id", &[10i32, 20, 30]).unwrap(),
        df_rt.series("score", &[1.0f64, 2.0, 3.0]).unwrap(),
    ])
    .unwrap();
    let indices = indices_rt.series("idx", &[2i32, 0]).unwrap();
    let expected_exec_stream = df.__debug_execution_domain_stream_value();
    let expected_exec_mr = df.__debug_execution_domain_mr_value();
    let expected_sync_stream = indices.as_column().__debug_execution_domain_stream_value();

    reset_exec_observation();
    let result = df.gather(&indices).unwrap();

    assert_eq!(result.__debug_execution_domain_id(), df.__debug_execution_domain_id());
    assert_eq!(result.__debug_execution_domain_mr_value(), expected_exec_mr);
    assert_eq!(last_exec_stream_value(), expected_exec_stream);
    assert_eq!(last_exec_mr_value(), expected_exec_mr);
    assert_eq!(last_sync_stream_value(), expected_sync_stream);

    reset_sync_observation();
    // Gather produces nullable output; use to_optional_vec.
    assert_eq!(result.series("id").unwrap().to_optional_vec::<i32>().unwrap(), vec![Some(30), Some(10)]);
    assert_eq!(result.series("score").unwrap().to_optional_vec::<f64>().unwrap(), vec![Some(3.0), Some(1.0)]);
    assert_eq!(last_sync_stream_value(), expected_exec_stream);
}

// ===========================================================================
// 2. filter_isin(&str, &Series)
// ===========================================================================

#[test]
fn test_filter_isin_with_series() {
    let df = sample_df(); // id: 10,20,30,40,50
    let values = crate::common::series::<i32>("vals", &[10, 30, 50]).unwrap();
    let result = df.filter_isin("id", &values).unwrap();
    assert_eq!(result.num_rows(), 3);
    assert_eq!(result.column_names(), vec!["id", "score"]);
    let mut ids = result.series("id").unwrap().to_vec_strict::<i32>().unwrap();
    ids.sort();
    assert_eq!(ids, vec![10, 30, 50]);
}

#[test]
fn test_filter_isin_with_series_succeeds_with_mixed_domains_and_uses_dataframe_provenance() {
    let df_rt = sample_runtime();
    let values_rt = sample_runtime();
    let df = DataFrame::from_series([df_rt.series("id", &[10i32, 20, 30]).unwrap()]).unwrap();
    let values = values_rt.series("vals", &[20i32, 30]).unwrap();
    let expected_exec_stream = df.__debug_execution_domain_stream_value();
    let expected_exec_mr = df.__debug_execution_domain_mr_value();
    let expected_sync_stream = values.as_column().__debug_execution_domain_stream_value();

    reset_exec_observation();
    let result = df.filter_isin("id", &values).unwrap();

    assert_eq!(result.__debug_execution_domain_id(), df.__debug_execution_domain_id());
    assert_eq!(result.__debug_execution_domain_mr_value(), expected_exec_mr);
    assert_eq!(last_exec_stream_value(), expected_exec_stream);
    assert_eq!(last_exec_mr_value(), expected_exec_mr);
    assert_eq!(last_sync_stream_value(), expected_sync_stream);

    reset_sync_observation();
    assert_eq!(result.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![20, 30]);
    assert_eq!(last_sync_stream_value(), expected_exec_stream);
}

#[test]
fn test_filter_isin_with_series_no_matches_preserves_schema() {
    let df = sample_df();
    let values = crate::common::series::<i32>("vals", &[99, 100]).unwrap();
    let result = df.filter_isin("id", &values).unwrap();

    assert_eq!(result.num_rows(), 0);
    assert_eq!(result.column_names(), vec!["id", "score"]);
}

#[test]
fn test_filter_isin_with_series_rejects_missing_column() {
    let df = sample_df();
    let values = crate::common::series::<i32>("vals", &[10]).unwrap();
    let err = df.filter_isin("nonexistent", &values).unwrap_err();

    assert_missing_column_error(&err, "nonexistent");
}

// ===========================================================================
// 3. fillna<T: CudfType>
// ===========================================================================

#[test]
fn test_fillna_with_plain_value() {
    let ids =
        crate::common::nullable_series::<i32>("id", &[10, 0, 30, 0, 50], &[true, false, true, false, true]).unwrap();
    let scores = crate::common::series::<f64>("score", &[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let df = DataFrame::from_series([ids, scores]).unwrap();

    let filled = df.fillna::<i32>("id", 42).unwrap();
    assert_eq!(filled.num_rows(), 5);
    assert_eq!(filled.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![10, 42, 30, 42, 50]);
    // Other columns unchanged.
    assert_eq!(filled.series("score").unwrap().to_vec_strict::<f64>().unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn test_fillna_with_scalar_still_works() {
    let ids =
        crate::common::nullable_series::<i32>("id", &[10, 0, 30, 0, 50], &[true, false, true, false, true]).unwrap();
    let scores = crate::common::series::<f64>("score", &[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let df = DataFrame::from_series([ids, scores]).unwrap();

    let fill = scalar(42i32).unwrap();
    let filled = df.fillna_with_scalar("id", &fill).unwrap();

    assert_eq!(filled.num_rows(), 5);
    assert_eq!(filled.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![10, 42, 30, 42, 50]);
    assert_eq!(filled.series("score").unwrap().to_vec_strict::<f64>().unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn test_fillna_rejects_missing_column() {
    let ids = crate::common::series::<i32>("id", &[10, 20, 30]).unwrap();
    let df = DataFrame::from_series([ids]).unwrap();
    let err = df.fillna::<i32>("nonexistent", 42).unwrap_err();

    assert_missing_column_error(&err, "nonexistent");
}

// ===========================================================================
// 4. factorize returns (Series, Series)
// ===========================================================================

#[test]
fn test_factorize_returns_series() {
    let df = sample_df(); // id: 10,20,30,40,50
    let (codes, uniques): (Series, Series) = df.factorize("id").unwrap();
    assert_eq!(codes.len(), 5);
    assert_eq!(uniques.len(), 5); // all values are unique
    // codes should have a name reflecting the source column
    assert!(codes.name().is_some());
    assert!(uniques.name().is_some());
}

// ===========================================================================
// 5. Advanced gather uses table_view() + low-level ops::copying
// ===========================================================================

#[test]
fn test_advanced_gather_still_works_via_table_view_and_ops() {
    let df = sample_df(); // id: 10,20,30,40,50
    let indices = column_from_slice(&[0i32, 2, 4]).unwrap();
    let table_view = df.table_view().unwrap();
    let result = copying::gather(table_view.table(), &indices.view().unwrap()).unwrap();
    let result = DataFrame::from_table(result, vec!["id".to_string(), "score".to_string()]).unwrap();

    assert_eq!(result.num_rows(), 3);
    // Gather produces nullable output; use to_optional_vec.
    assert_eq!(result.series("id").unwrap().to_optional_vec::<i32>().unwrap(), vec![Some(10), Some(30), Some(50)]);
    assert_eq!(
        result.series("score").unwrap().to_optional_vec::<f64>().unwrap(),
        vec![Some(1.0), Some(3.0), Some(5.0)]
    );
    // Schema preserved.
    assert_eq!(result.column_names(), vec!["id", "score"]);
}

// ===========================================================================
// 6. Advanced filter_isin uses table_view() + low-level ops
// ===========================================================================

#[test]
fn test_advanced_filter_isin_still_works_via_table_view_and_ops() {
    let df = sample_df(); // id: 10,20,30,40,50
    let values = column_from_slice(&[10i32, 30, 50]).unwrap();
    let values_view = values.view().unwrap();
    let ids = df.series("id").unwrap();
    let ids_view = ids.as_column_view().unwrap();
    let mask = search::contains(&values_view, &ids_view).unwrap();
    let mask_view = mask.view().unwrap();
    let table_view = df.table_view().unwrap();
    let result = low_level_filter::apply_boolean_mask(table_view.table(), &mask_view).unwrap();
    let result = DataFrame::from_table(result, vec!["id".to_string(), "score".to_string()]).unwrap();

    assert_eq!(result.num_rows(), 3);
    let mut ids = result.series("id").unwrap().to_vec_strict::<i32>().unwrap();
    ids.sort();
    assert_eq!(ids, vec![10, 30, 50]);
    // Schema preserved.
    assert_eq!(result.column_names(), vec!["id", "score"]);
}
