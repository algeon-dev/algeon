//! Tests for high-level primary DataFrame signatures.
//!
//! These tests cover the current high-level signatures that accept `Series`
//! or plain Rust values and preserve explicit lower-level entry points where
//! needed.
//!
//! # Planned changes
//!
//! | Method                   | Current signature                          | Target signature                           |
//! |--------------------------|--------------------------------------------|--------------------------------------------|
//! | `gather`                 | `&Series` (primary high-level API)         | `gather_by_column_view` for `&ColumnView`  |
//! | `filter_isin`            | `&Series` (primary high-level API)         | `filter_isin_by_column_view` for `&ColumnView` |
//! | `fillna`                 | `T: CudfType` (primary high-level API)     | `fillna_with_scalar` for `&Scalar`         |
//! | `factorize`              | Returns `(Column, Column)`                 | Returns `(Series, Series)`                 |

use cudf::{Column, DataFrame, Series};

use crate::common::{
    last_exec_mr_value, last_exec_stream_value, last_sync_stream_value, reset_exec_observation, reset_sync_observation,
    sample_runtime,
};

// ===========================================================================
// Helpers
// ===========================================================================

/// Build a 5-row DataFrame with id (i32) and score (f64) columns.
fn sample_df() -> DataFrame {
    let ids = Series::new::<i32>("id", &[10, 20, 30, 40, 50]).unwrap();
    let scores = Series::new::<f64>("score", &[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    DataFrame::from_columns([("id", ids), ("score", scores)]).unwrap()
}

// ===========================================================================
// 1. gather(&Series) — primary high-level API (was gather_by_series)
// ===========================================================================

#[test]
fn test_gather_with_series() {
    let df = sample_df(); // id: 10,20,30,40,50
    let indices = Series::new::<i32>("idx", &[0, 2, 4]).unwrap();
    let result = df.gather(&indices).unwrap();
    assert_eq!(result.num_rows(), 3);
    assert_eq!(result.series("id").unwrap().to_vec::<i32>().unwrap(), vec![10, 30, 50]);
    assert_eq!(result.series("score").unwrap().to_vec::<f64>().unwrap(), vec![1.0, 3.0, 5.0]);
}

#[test]
fn test_gather_with_series_succeeds_with_mixed_domains_and_uses_dataframe_provenance() {
    let df_rt = sample_runtime();
    let indices_rt = sample_runtime();
    let df = DataFrame::from_series([
        ("id", Series::new_with_runtime(&df_rt, "id", &[10i32, 20, 30]).unwrap()),
        ("score", Series::new_with_runtime(&df_rt, "score", &[1.0f64, 2.0, 3.0]).unwrap()),
    ])
    .unwrap();
    let indices = Series::new_with_runtime(&indices_rt, "idx", &[2i32, 0]).unwrap();
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
    assert_eq!(result.series("id").unwrap().to_vec::<i32>().unwrap(), vec![30, 10]);
    assert_eq!(result.series("score").unwrap().to_vec::<f64>().unwrap(), vec![3.0, 1.0]);
    assert_eq!(last_sync_stream_value(), expected_exec_stream);
}

// ===========================================================================
// 2. filter_isin(&str, &Series) — primary high-level API (was filter_isin_series)
// ===========================================================================

#[test]
fn test_filter_isin_with_series() {
    let df = sample_df(); // id: 10,20,30,40,50
    let values = Series::new::<i32>("vals", &[10, 30, 50]).unwrap();
    let result = df.filter_isin("id", &values).unwrap();
    assert_eq!(result.num_rows(), 3);
    assert_eq!(result.column_names(), vec!["id", "score"]);
    let mut ids = result.series("id").unwrap().to_vec::<i32>().unwrap();
    ids.sort();
    assert_eq!(ids, vec![10, 30, 50]);
}

#[test]
fn test_filter_isin_with_series_succeeds_with_mixed_domains_and_uses_dataframe_provenance() {
    let df_rt = sample_runtime();
    let values_rt = sample_runtime();
    let df =
        DataFrame::from_series([("id", Series::new_with_runtime(&df_rt, "id", &[10i32, 20, 30]).unwrap())]).unwrap();
    let values = Series::new_with_runtime(&values_rt, "vals", &[20i32, 30]).unwrap();
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
    assert_eq!(result.series("id").unwrap().to_vec::<i32>().unwrap(), vec![20, 30]);
    assert_eq!(last_sync_stream_value(), expected_exec_stream);
}

#[test]
fn test_filter_isin_with_series_no_matches_preserves_schema() {
    let df = sample_df();
    let values = Series::new::<i32>("vals", &[99, 100]).unwrap();
    let result = df.filter_isin("id", &values).unwrap();

    assert_eq!(result.num_rows(), 0);
    assert_eq!(result.column_names(), vec!["id", "score"]);
}

#[test]
fn test_filter_isin_with_series_rejects_missing_column() {
    let df = sample_df();
    let values = Series::new::<i32>("vals", &[10]).unwrap();
    let err = df.filter_isin("nonexistent", &values).unwrap_err();

    assert!(err.to_string().contains("nonexistent"));
}

// ===========================================================================
// 3. fillna<T: CudfType> — primary high-level API (was fillna_scalar)
// ===========================================================================

#[test]
fn test_fillna_scalar_plain_value() {
    let ids = Series::new_nullable::<i32>("id", &[10, 0, 30, 0, 50], &[true, false, true, false, true]).unwrap();
    let scores = Series::new::<f64>("score", &[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let df = DataFrame::from_columns([("id", ids), ("score", scores)]).unwrap();

    let filled = df.fillna::<i32>("id", 42).unwrap();
    assert_eq!(filled.num_rows(), 5);
    assert_eq!(filled.series("id").unwrap().to_vec::<i32>().unwrap(), vec![10, 42, 30, 42, 50]);
    // Other columns unchanged.
    assert_eq!(filled.series("score").unwrap().to_vec::<f64>().unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn test_fillna_with_scalar_still_works() {
    let ids = Series::new_nullable::<i32>("id", &[10, 0, 30, 0, 50], &[true, false, true, false, true]).unwrap();
    let scores = Series::new::<f64>("score", &[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let df = DataFrame::from_columns([("id", ids), ("score", scores)]).unwrap();

    let fill = cudf::Scalar::from_value(42i32).unwrap();
    let filled = df.fillna_with_scalar("id", &fill).unwrap();

    assert_eq!(filled.num_rows(), 5);
    assert_eq!(filled.series("id").unwrap().to_vec::<i32>().unwrap(), vec![10, 42, 30, 42, 50]);
    assert_eq!(filled.series("score").unwrap().to_vec::<f64>().unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn test_fillna_rejects_missing_column() {
    let ids = Series::new::<i32>("id", &[10, 20, 30]).unwrap();
    let df = DataFrame::from_columns([("id", ids)]).unwrap();
    let err = df.fillna::<i32>("nonexistent", 42).unwrap_err();

    assert!(err.to_string().contains("nonexistent"));
}

// ===========================================================================
// 4. factorize returns (Series, Series) — implemented (P2-E)
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
// 5. gather_by_column_view(&ColumnView) — advanced entry point
// ===========================================================================

#[test]
fn test_gather_columnview_still_works() {
    let df = sample_df(); // id: 10,20,30,40,50
    let indices = Column::from_slice(&[0i32, 2, 4]).unwrap();
    let result = df.gather_by_column_view(&indices.view().unwrap()).unwrap();

    assert_eq!(result.num_rows(), 3);
    assert_eq!(result.series("id").unwrap().to_vec::<i32>().unwrap(), vec![10, 30, 50]);
    assert_eq!(result.series("score").unwrap().to_vec::<f64>().unwrap(), vec![1.0, 3.0, 5.0]);
    // Schema preserved.
    assert_eq!(result.column_names(), vec!["id", "score"]);
}

// ===========================================================================
// 6. filter_isin_by_column_view(&str, &ColumnView) — advanced entry point
// ===========================================================================

#[test]
fn test_filter_isin_columnview_still_works() {
    let df = sample_df(); // id: 10,20,30,40,50
    let values = Column::from_slice(&[10i32, 30, 50]).unwrap();
    let values_view = values.view().unwrap();
    let result = df.filter_isin_by_column_view("id", &values_view).unwrap();

    assert_eq!(result.num_rows(), 3);
    let mut ids = result.series("id").unwrap().to_vec::<i32>().unwrap();
    ids.sort();
    assert_eq!(ids, vec![10, 30, 50]);
    // Schema preserved.
    assert_eq!(result.column_names(), vec!["id", "score"]);
}
