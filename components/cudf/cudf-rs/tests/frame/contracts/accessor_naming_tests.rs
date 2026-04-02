//! Accessor naming consistency tests
//!
//! Validates the accessor naming scheme:
//! - `DataFrame::series()` — returns owned `Series`
//! - `DataFrame::get()` — probe access, returns `None` on miss
//! - `DataFrame::contains()` — existence check
//! - `DataFrame::iter_series()` — schema-ordered iteration
//! - `Series::as_column()` — returns `&Column`
//! - `Series::into_inner()` — consumes and returns `Arc<Column>`
//!
//! This file covers the current accessor naming surface and keeps the
//! high-level/low-level accessor boundary explicit.

use cudf::{DataFrame, Series, TypeId};

// ===========================================================================
// Helpers
// ===========================================================================

fn sample_df() -> DataFrame {
    let ids = Series::new::<i32>("id", &[1, 2, 3]).unwrap();
    let scores = Series::new::<f64>("score", &[10.0, 20.0, 30.0]).unwrap();
    DataFrame::from_columns([("id", ids), ("score", scores)]).unwrap()
}

// ===========================================================================
// 1. DataFrame::series() — hit
// ===========================================================================

/// `series()` returns a `Series` for an existing column.
#[test]
fn test_dataframe_series_hit() {
    let df = sample_df();
    let s = df.series("id").unwrap();

    assert_eq!(s.name(), Some("id"));
    assert_eq!(s.len(), 3);
    assert_eq!(s.dtype().unwrap().id(), TypeId::Int32);
    assert_eq!(s.to_vec::<i32>().unwrap(), vec![1, 2, 3]);
}

// ===========================================================================
// 2. DataFrame::series() — miss
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
// 3. Two series() calls share the same device buffer
// ===========================================================================

/// Two `series()` calls return `Series` backed by the same GPU buffer.
#[test]
fn test_dataframe_series_share_device_data() {
    let df = sample_df();
    let ptr1 = df.series("id").unwrap().as_column_view().unwrap().device_data_ptr().unwrap();
    let ptr2 = df.series("id").unwrap().as_column_view().unwrap().device_data_ptr().unwrap();

    assert_eq!(ptr1, ptr2, "two series() calls should share the same GPU buffer");
}

// ===========================================================================
// 4. DataFrame::get() — probe access
// ===========================================================================

/// `get()` returns `None` on a missing column name.
#[test]
fn test_dataframe_get_miss() {
    let df = sample_df();
    let result = df.get("nonexistent");
    assert!(result.is_none());
}

/// `get()` returns `Some(Series)` for an existing column.
#[test]
fn test_dataframe_get_hit() {
    let df = sample_df();
    let s = df.get("id").unwrap();
    assert_eq!(s.name(), Some("id"));
    assert_eq!(s.to_vec::<i32>().unwrap(), vec![1, 2, 3]);
}

// ===========================================================================
// 5. DataFrame::contains()
// ===========================================================================

/// `contains()` returns `true` for existing columns, `false` otherwise.
#[test]
fn test_dataframe_contains() {
    let df = sample_df();

    assert!(df.contains("id"));
    assert!(df.contains("score"));
    assert!(!df.contains("nonexistent"));
}

// ===========================================================================
// 6. DataFrame::iter_series() — schema-ordered iteration
// ===========================================================================

/// `iter_series()` yields Series in schema (insertion) order.
#[test]
fn test_dataframe_iter_series_preserves_schema_order() {
    let df = sample_df(); // columns: "id", "score"
    let names: Vec<_> = df.iter_series().map(|s| s.name().unwrap().to_string()).collect();

    assert_eq!(names, vec!["id", "score"]);
}

/// `iter_series()` assigns the DataFrame column name to each Series.
#[test]
fn test_dataframe_iter_series_assigns_column_names() {
    let df = sample_df();
    let items: Vec<_> = df.iter_series().collect::<Vec<_>>();

    assert_eq!(items[0].name(), Some("id"));
    assert_eq!(items[0].to_vec::<i32>().unwrap(), vec![1, 2, 3]);
    assert_eq!(items[1].name(), Some("score"));
    assert_eq!(items[1].to_vec::<f64>().unwrap(), vec![10.0, 20.0, 30.0]);
}

/// `iter_series()` shares the underlying device buffer (Arc-sharing, no GPU copy).
#[test]
fn test_dataframe_iter_series_shares_device_buffers() {
    let df = sample_df();
    let items: Vec<_> = df.iter_series().collect::<Vec<_>>();

    let iter_ptr = items[0].as_column_view().unwrap().device_data_ptr().unwrap();
    let direct_ptr = df.series("id").unwrap().as_column_view().unwrap().device_data_ptr().unwrap();

    assert_eq!(iter_ptr, direct_ptr, "iter_series() should share the same GPU buffer as series()");
}

// ===========================================================================
// 7. Series::as_column() — explicit "drop to core"
// ===========================================================================

/// `as_column()` returns a `&Column` reference from a `Series`.
#[test]
fn test_series_as_column() {
    let s = Series::new::<i32>("x", &[10, 20, 30]).unwrap();
    let col: &cudf::Column = s.as_column();

    // The column should have the same length as the series.
    let view = col.view().unwrap();
    assert_eq!(view.len().unwrap(), 3);
    assert_eq!(view.to_vec::<i32>().unwrap(), vec![10, 20, 30]);
}

// ===========================================================================
// 7. Series::into_inner() — consumes and returns Arc<Column>
// ===========================================================================

/// `into_inner()` consumes the `Series` and returns the underlying `Arc<Column>`.
#[test]
fn test_series_into_inner() {
    let s = Series::new::<i32>("x", &[10, 20, 30]).unwrap();
    let arc: std::sync::Arc<cudf::Column> = s.into_inner();

    // Should be a valid Column with the same data.
    let view = arc.view().unwrap();
    assert_eq!(view.len().unwrap(), 3);
    assert_eq!(view.to_vec::<i32>().unwrap(), vec![10, 20, 30]);
}

// ===========================================================================
// 8. DataFrame::series() works for borrowing pattern
// ===========================================================================

/// `DataFrame::series()` returns a `Series` for borrowing.
#[test]
fn test_series_access_works() {
    let df = sample_df();
    let s = df.series("score").unwrap();

    assert_eq!(s.name(), Some("score"));
    assert_eq!(s.len(), 3);
    assert_eq!(s.dtype().unwrap().id(), TypeId::Float64);
    assert_eq!(s.to_vec::<f64>().unwrap(), vec![10.0, 20.0, 30.0]);
}
