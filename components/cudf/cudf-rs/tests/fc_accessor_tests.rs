//! FC-ACCESSOR: Accessor naming cleanup tests
//!
//! Validates that:
//! - `series()` is the canonical owned accessor (returns `Series`)
//! - `column_view()` is the explicit borrow accessor (returns `SeriesView`)
//! - `column()` still works but is deprecated (backward compat)
//! - `column_view()` and `series()` refer to the same GPU data

use cudf::{dataframe::DataFrame, series::Series, types::TypeId};

// ===========================================================================
// Helpers
// ===========================================================================

fn sample_df() -> DataFrame {
    let ids = Series::new::<i32>("id", &[1, 2, 3]).unwrap();
    let scores = Series::new::<f64>("score", &[10.0, 20.0, 30.0]).unwrap();
    DataFrame::from_columns([("id", ids), ("score", scores)]).unwrap()
}

// ===========================================================================
// 1. series() — canonical owned accessor
// ===========================================================================

/// `series()` returns an owned `Series` — the canonical façade accessor.
#[test]
fn series_returns_owned_series() {
    let df = sample_df();
    let s = df.series("id").unwrap();

    assert_eq!(s.name(), Some("id"));
    assert_eq!(s.len(), 3);
    assert_eq!(s.dtype().unwrap().id(), TypeId::Int32);
}

// ===========================================================================
// 2. column_view() — explicit borrow accessor
// ===========================================================================

/// `column_view()` returns a borrowed `SeriesView` — explicit borrow accessor.
#[test]
fn column_view_returns_series_view() {
    let df = sample_df();
    let sv = df.column_view("score").unwrap();

    assert_eq!(sv.name(), Some("score"));
    assert_eq!(sv.len().unwrap(), 3);
    assert_eq!(sv.data_type().unwrap().id(), TypeId::Float64);
    assert_eq!(sv.to_vec::<f64>().unwrap(), vec![10.0, 20.0, 30.0]);
}

// ===========================================================================
// 3. column_view() same device pointer as series() (zero-copy)
// ===========================================================================

/// `column_view()` and `series()` point to the same GPU buffer.
#[test]
fn column_view_same_device_ptr_as_series() {
    let df = sample_df();
    let sv = df.column_view("id").unwrap();
    let view_ptr = sv.device_data_ptr().unwrap();

    let s = df.series("id").unwrap();
    let series_ptr = s.column_view().unwrap().device_data_ptr().unwrap();

    assert_eq!(view_ptr, series_ptr, "column_view() and series() must share GPU buffer");
}

// ===========================================================================
// 4. column() still works (deprecated backward compat)
// ===========================================================================

/// `column()` still compiles and returns correct results.
/// It is deprecated but must remain functional.
#[allow(deprecated)]
#[test]
fn deprecated_column_still_works() {
    let df = sample_df();
    let sv = df.column("id").unwrap();

    assert_eq!(sv.name(), Some("id"));
    assert_eq!(sv.to_vec::<i32>().unwrap(), vec![1, 2, 3]);
}

// ===========================================================================
// 5. column_view() error on missing column name
// ===========================================================================

/// `column_view()` returns an error for a non-existent column.
#[test]
fn column_view_error_on_missing_column() {
    let df = sample_df();
    let result = df.column_view("nonexistent");

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("not found"), "error message should mention 'not found': {err}");
}
