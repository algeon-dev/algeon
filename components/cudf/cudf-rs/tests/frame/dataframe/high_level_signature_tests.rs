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
//! | `gather`                 | `&Series` (facade primary)                 | `gather_by_column_view` for `&ColumnView`  |
//! | `filter_isin`            | `&Series` (facade primary)                 | `filter_isin_by_column_view` for `&ColumnView` |
//! | `fillna`                 | `T: CudfType` (facade primary)             | `fillna_with_scalar` for `&Scalar`         |
//! | `factorize`              | Returns `(Column, Column)`                 | Returns `(Series, Series)`                 |

use cudf::{Column, DataFrame, Series};

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
// 1. gather(&Series) — facade primary (was gather_by_series)
// ===========================================================================

#[test]
fn test_gather_with_series() {
    let df = sample_df(); // id: 10,20,30,40,50
    let indices = Series::new::<i32>("idx", &[0, 2, 4]).unwrap();
    let result = df.gather(&indices).unwrap();
    assert_eq!(result.num_rows(), 3);
    assert_eq!(result.series_view("id").unwrap().to_vec::<i32>().unwrap(), vec![10, 30, 50]);
    assert_eq!(result.series_view("score").unwrap().to_vec::<f64>().unwrap(), vec![1.0, 3.0, 5.0]);
}

// ===========================================================================
// 2. filter_isin(&str, &Series) — facade primary (was filter_isin_series)
// ===========================================================================

#[test]
fn test_filter_isin_with_series() {
    let df = sample_df(); // id: 10,20,30,40,50
    let values = Series::new::<i32>("vals", &[10, 30, 50]).unwrap();
    let result = df.filter_isin("id", &values).unwrap();
    assert_eq!(result.num_rows(), 3);
    assert_eq!(result.column_names(), vec!["id", "score"]);
    let mut ids = result.series_view("id").unwrap().to_vec::<i32>().unwrap();
    ids.sort();
    assert_eq!(ids, vec![10, 30, 50]);
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
// 3. fillna<T: CudfType> — facade primary (was fillna_scalar)
// ===========================================================================

#[test]
fn test_fillna_scalar_plain_value() {
    let ids = Series::new_nullable::<i32>("id", &[10, 0, 30, 0, 50], &[true, false, true, false, true]).unwrap();
    let scores = Series::new::<f64>("score", &[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let df = DataFrame::from_columns([("id", ids), ("score", scores)]).unwrap();

    let filled = df.fillna::<i32>("id", 42).unwrap();
    assert_eq!(filled.num_rows(), 5);
    assert_eq!(filled.series_view("id").unwrap().to_vec::<i32>().unwrap(), vec![10, 42, 30, 42, 50]);
    // Other columns unchanged.
    assert_eq!(filled.series_view("score").unwrap().to_vec::<f64>().unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn test_fillna_with_scalar_still_works() {
    let ids = Series::new_nullable::<i32>("id", &[10, 0, 30, 0, 50], &[true, false, true, false, true]).unwrap();
    let scores = Series::new::<f64>("score", &[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let df = DataFrame::from_columns([("id", ids), ("score", scores)]).unwrap();

    let fill = cudf::Scalar::from_value(42i32).unwrap();
    let filled = df.fillna_with_scalar("id", &fill).unwrap();

    assert_eq!(filled.num_rows(), 5);
    assert_eq!(filled.series_view("id").unwrap().to_vec::<i32>().unwrap(), vec![10, 42, 30, 42, 50]);
    assert_eq!(filled.series_view("score").unwrap().to_vec::<f64>().unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
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
    assert_eq!(result.series_view("id").unwrap().to_vec::<i32>().unwrap(), vec![10, 30, 50]);
    assert_eq!(result.series_view("score").unwrap().to_vec::<f64>().unwrap(), vec![1.0, 3.0, 5.0]);
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
    let mut ids = result.series_view("id").unwrap().to_vec::<i32>().unwrap();
    ids.sort();
    assert_eq!(ids, vec![10, 30, 50]);
    // Schema preserved.
    assert_eq!(result.column_names(), vec!["id", "score"]);
}
