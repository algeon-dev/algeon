//! P3-T: Accessor naming consistency tests (TDD)
//!
//! Validates the target accessor naming scheme:
//! - `DataFrame::series_view()` — replaces `column()`, returns `SeriesView`
//! - `Series::as_column()` — replaces `column()`, returns `&Column`
//! - `Series::into_inner()` — replaces `column_arc()`, consumes and returns `Arc<Column>`
//! - `DataFrameView::series_view()` — returns `Err` on miss (consistent with DataFrame)
//! - `DataFrameView::contains()` — returns `bool`
//!
//! This file covers the current accessor naming surface and keeps the
//! high-level/low-level accessor boundary explicit.

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
// 1. DataFrame::series_view() — hit
// ===========================================================================

/// `series_view()` returns a `SeriesView` for an existing column.
#[test]
fn test_dataframe_series_view_hit() {
    let df = sample_df();
    let sv = df.series_view("id").unwrap();

    assert_eq!(sv.name(), Some("id"));
    assert_eq!(sv.len().unwrap(), 3);
    assert_eq!(sv.data_type().unwrap().id(), TypeId::Int32);
    assert_eq!(sv.to_vec::<i32>().unwrap(), vec![1, 2, 3]);
}

// ===========================================================================
// 2. DataFrame::series_view() — miss
// ===========================================================================

/// `series_view()` returns `Err` when the column does not exist.
#[test]
fn test_dataframe_series_view_miss() {
    let df = sample_df();
    let result = df.series_view("nonexistent");

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("not found"), "error should mention 'not found': {err}");
}

// ===========================================================================
// 3. DataFrame::series() — hit (already exists)
// ===========================================================================

/// `series()` returns an owned `Series` for an existing column.
#[test]
fn test_dataframe_series_hit() {
    let df = sample_df();
    let s = df.series("id").unwrap();

    assert_eq!(s.name(), Some("id"));
    assert_eq!(s.len(), 3);
    assert_eq!(s.dtype().unwrap().id(), TypeId::Int32);

    let view = s.view().unwrap();
    assert_eq!(view.to_vec::<i32>().unwrap(), vec![1, 2, 3]);
}

/// `series()` and `series_view()` refer to the same device buffer.
#[test]
fn test_dataframe_series_and_series_view_share_device_data() {
    let df = sample_df();
    let view_ptr = df.series_view("id").unwrap().device_data_ptr().unwrap();
    let series_ptr = df.series("id").unwrap().as_column_view().unwrap().device_data_ptr().unwrap();

    assert_eq!(view_ptr, series_ptr, "owned and borrowed accessors should share the same GPU buffer");
}

// ===========================================================================
// 4. DataFrame::series() — miss (already exists)
// ===========================================================================

/// `series()` returns `Err` when the column does not exist.
#[test]
fn test_dataframe_series_miss() {
    let df = sample_df();
    let result = df.series("nonexistent");

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("not found"), "error should mention 'not found': {err}");
}

// ===========================================================================
// 5. Series::as_column() — explicit "drop to core"
// ===========================================================================

/// `as_column()` returns a `&Column` reference from a `Series`.
#[test]
fn test_series_as_column() {
    let s = Series::new::<i32>("x", &[10, 20, 30]).unwrap();
    let col: &cudf::column::Column = s.as_column();

    // The column should have the same length as the series.
    let view = col.view().unwrap();
    assert_eq!(view.len().unwrap(), 3);
    assert_eq!(view.to_vec::<i32>().unwrap(), vec![10, 20, 30]);
}

// ===========================================================================
// 6. Series::into_inner() — consumes and returns Arc<Column>
// ===========================================================================

/// `into_inner()` consumes the `Series` and returns the underlying `Arc<Column>`.
#[test]
fn test_series_into_inner() {
    let s = Series::new::<i32>("x", &[10, 20, 30]).unwrap();
    let arc: std::sync::Arc<cudf::column::Column> = s.into_inner();

    // Should be a valid Column with the same data.
    let view = arc.view().unwrap();
    assert_eq!(view.len().unwrap(), 3);
    assert_eq!(view.to_vec::<i32>().unwrap(), vec![10, 20, 30]);
}

// ===========================================================================
// 7. DataFrameView::series_view() — miss returns Err
// ===========================================================================

/// `DataFrameView::series_view()` returns `Err` on a missing column name,
/// consistent with `DataFrame::series_view()`.
#[test]
fn test_dataframe_view_series_view_miss() {
    let df = sample_df();
    let view = df.view();
    let result = view.series_view("nonexistent");

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("not found"), "error should mention 'not found': {err}");
}

// ===========================================================================
// 8. DataFrameView::contains()
// ===========================================================================

/// `DataFrameView::contains()` returns `true` for existing columns, `false` otherwise.
#[test]
fn test_dataframe_view_contains() {
    let df = sample_df();
    let view = df.view();

    assert!(view.contains("id"));
    assert!(view.contains("score"));
    assert!(!view.contains("nonexistent"));
}

// ===========================================================================
// 10. DataFrame::series_view() works
// ===========================================================================

/// `DataFrame::series_view()` returns a `SeriesView` for borrowing.
#[test]
fn test_series_view_works() {
    let df = sample_df();
    let sv = df.series_view("score").unwrap();

    assert_eq!(sv.name(), Some("score"));
    assert_eq!(sv.len().unwrap(), 3);
    assert_eq!(sv.data_type().unwrap().id(), TypeId::Float64);
    assert_eq!(sv.to_vec::<f64>().unwrap(), vec![10.0, 20.0, 30.0]);
}
