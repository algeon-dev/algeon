//! Tests for explicit copy-semantic naming on DataFrame.
//!
//! These tests verify that the new explicitly-named conversion methods
//! (`to_table_copy`, `try_into_table`, `temp_table_view`) preserve the intended
//! deep-copy vs zero-copy behavior of the corresponding conversion paths.
//!
//! All tests require a GPU.

use cudf::{Column, DataFrame, Series, TypeId};

// ===========================================================================
// Helpers
// ===========================================================================

/// Build a 3-row, 2-column DataFrame.
fn sample_df() -> DataFrame {
    let ids = Series::new::<i32>("id", &[1, 2, 3]).unwrap();
    let scores = Series::new::<f64>("score", &[10.0, 20.0, 30.0]).unwrap();
    DataFrame::from_columns([("id", ids), ("score", scores)]).unwrap()
}

/// Get device data pointer for a named column in a DataFrame.
fn device_ptr(df: &DataFrame, name: &str) -> *const std::ffi::c_void {
    df.series_view(name).unwrap().device_data_ptr().unwrap()
}

// ===========================================================================
// 1. to_table_copy() — deep-copies all columns
// ===========================================================================

/// `to_table_copy()` produces a table with independent GPU memory.
#[test]
fn to_table_copy_deep_copies() {
    let df = sample_df();
    let table = df.to_table_copy().unwrap();

    let tv = table.view().unwrap();
    assert_eq!(tv.num_columns().unwrap(), 2);
    assert_eq!(tv.num_rows().unwrap(), 3);

    // The table's column 0 must have a DIFFERENT device pointer (deep copy).
    let table_ptr = tv.column(0).unwrap().device_data_ptr().unwrap();
    let df_ptr = device_ptr(&df, "id");
    assert_ne!(table_ptr, df_ptr, "to_table_copy() must deep-copy; pointers should differ");

    // Values should still match.
    assert_eq!(tv.column(0).unwrap().to_vec::<i32>().unwrap(), vec![1, 2, 3]);
    assert_eq!(tv.column(1).unwrap().to_vec::<f64>().unwrap(), vec![10.0, 20.0, 30.0]);
}

/// `to_table_copy()` does not consume the DataFrame.
#[test]
fn to_table_copy_source_still_usable() {
    let df = sample_df();
    let _table = df.to_table_copy().unwrap();

    assert_eq!(df.series_view("id").unwrap().to_vec::<i32>().unwrap(), vec![1, 2, 3]);
    assert_eq!(df.series_view("score").unwrap().to_vec::<f64>().unwrap(), vec![10.0, 20.0, 30.0]);
}

// ===========================================================================
// 2. try_into_table() — zero-copy when uniquely owned
// ===========================================================================

/// When the DataFrame is the sole owner of all columns (Arc strong count == 1),
/// `try_into_table()` unwraps without deep-copying.
#[test]
fn try_into_table_unique_owner_no_copy() {
    let df = DataFrame::new(vec!["a".into()], vec![Column::from_slice(&[1i32, 2, 3]).unwrap()]).unwrap();

    let original_ptr = device_ptr(&df, "a");
    let (table, names) = df.try_into_table().unwrap();

    assert_eq!(names, vec!["a"]);
    let tv = table.view().unwrap();
    let table_ptr = tv.column(0).unwrap().device_data_ptr().unwrap();

    assert_eq!(original_ptr, table_ptr, "try_into_table() with unique Arc should transfer without copy");
}

// ===========================================================================
// 3. try_into_table() — deep-copies when shared
// ===========================================================================

/// When columns are shared (Arc strong count > 1), `try_into_table()` must
/// deep-copy those columns.
#[test]
fn try_into_table_shared_columns_deep_copies() {
    let df = sample_df();

    // Hold an external reference to keep the Arc refcount > 1.
    let external_series = df.series("id").unwrap();
    let external_ptr = external_series.as_column_view().unwrap().device_data_ptr().unwrap();

    let original_ptr = device_ptr(&df, "id");
    assert_eq!(original_ptr, external_ptr, "series shares same Arc");

    let (table, names) = df.try_into_table().unwrap();
    assert_eq!(names, vec!["id", "score"]);

    let tv = table.view().unwrap();
    let table_ptr = tv.column(0).unwrap().device_data_ptr().unwrap();

    assert_ne!(table_ptr, external_ptr, "try_into_table() with shared Arc must deep-copy; pointers should differ");

    // External series is still valid.
    assert_eq!(external_series.as_column_view().unwrap().to_vec::<i32>().unwrap(), vec![1, 2, 3],);
}

/// `try_into_table()` preserves data values.
#[test]
fn try_into_table_values_preserved() {
    let df = sample_df();
    let (table, names) = df.try_into_table().unwrap();

    assert_eq!(names, vec!["id", "score"]);
    let tv = table.view().unwrap();
    assert_eq!(tv.column(0).unwrap().to_vec::<i32>().unwrap(), vec![1, 2, 3]);
    assert_eq!(tv.column(1).unwrap().to_vec::<f64>().unwrap(), vec![10.0, 20.0, 30.0]);
}

// ===========================================================================
// 4. temp_table_view() — truly zero-copy borrowed view
// ===========================================================================

/// `temp_table_view()` returns a `TempTableView` whose `table()` points to
/// the same GPU data — no copy.
#[test]
fn temp_table_view_zero_copy() {
    let df = sample_df();
    let ttv = df.temp_table_view().unwrap();
    let tv = ttv.table();

    assert_eq!(tv.num_columns().unwrap(), 2);
    assert_eq!(tv.num_rows().unwrap(), 3);

    // Device pointers must match — no copy.
    let tv_ptr = tv.column(0).unwrap().device_data_ptr().unwrap();
    let df_ptr = device_ptr(&df, "id");
    assert_eq!(tv_ptr, df_ptr, "temp_table_view() must be zero-copy");
}

/// `temp_table_view()` preserves column order.
#[test]
fn temp_table_view_preserves_column_order() {
    let df = sample_df();
    let ttv = df.temp_table_view().unwrap();
    let tv = ttv.table();

    // Column 0 = "id" (Int32), Column 1 = "score" (Float64)
    assert_eq!(tv.column(0).unwrap().data_type().unwrap().id(), TypeId::Int32);
    assert_eq!(tv.column(1).unwrap().data_type().unwrap().id(), TypeId::Float64);

    assert_eq!(tv.column(0).unwrap().to_vec::<i32>().unwrap(), vec![1, 2, 3]);
    assert_eq!(tv.column(1).unwrap().to_vec::<f64>().unwrap(), vec![10.0, 20.0, 30.0]);
}

// ===========================================================================
// 5. Renamed methods produce correct results
// ===========================================================================

/// `to_table_copy()` produces correct results.
#[test]
fn to_table_copy_produces_correct_results() {
    let df = sample_df();
    let table = df.to_table_copy().unwrap();

    let tv = table.view().unwrap();
    assert_eq!(tv.num_columns().unwrap(), 2);
    assert_eq!(tv.num_rows().unwrap(), 3);
    assert_eq!(tv.column(0).unwrap().to_vec::<i32>().unwrap(), vec![1, 2, 3]);
}

/// `try_into_table()` produces correct results.
#[test]
fn try_into_table_produces_correct_results() {
    let df = sample_df();
    let (table, names) = df.try_into_table().unwrap();

    assert_eq!(names, vec!["id", "score"]);
    let tv = table.view().unwrap();
    assert_eq!(tv.column(0).unwrap().to_vec::<i32>().unwrap(), vec![1, 2, 3]);
}
