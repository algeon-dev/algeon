//! Integration tests for `SeriesView` — a named borrow of a GPU column.
//!
//! All tests require a GPU.

use cudf::{DataFrame, column::Column, series::Series, types::TypeId};

// ---------------------------------------------------------------------------
// SeriesView from DataFrame::series_view() carries name
// ---------------------------------------------------------------------------

#[test]
fn view_carries_name() {
    let ids = Series::new::<i32>("x", &[1, 2, 3]).unwrap();
    let df = DataFrame::from_columns([("x", ids)]).unwrap();

    let sv = df.series_view("x").unwrap();
    assert_eq!(sv.name(), Some("x"));
}

// ---------------------------------------------------------------------------
// Deref coercion: data_type(), len(), to_vec(), device_data_ptr()
// ---------------------------------------------------------------------------

#[test]
fn deref_data_type() {
    let s = Series::new::<i32>("a", &[10, 20]).unwrap();
    let df = DataFrame::from_columns([("a", s)]).unwrap();

    let sv = df.series_view("a").unwrap();
    // data_type() is on ColumnView — accessed via Deref
    assert_eq!(sv.data_type().unwrap().id(), TypeId::Int32);
}

#[test]
fn deref_len() {
    let s = Series::new::<f64>("v", &[1.0, 2.0, 3.0]).unwrap();
    let df = DataFrame::from_columns([("v", s)]).unwrap();

    let sv = df.series_view("v").unwrap();
    assert_eq!(sv.len().unwrap(), 3);
}

#[test]
fn deref_to_vec() {
    let s = Series::new::<i32>("nums", &[10, 20, 30]).unwrap();
    let df = DataFrame::from_columns([("nums", s)]).unwrap();

    let sv = df.series_view("nums").unwrap();
    let vals: Vec<i32> = sv.to_vec().unwrap();
    assert_eq!(vals, vec![10, 20, 30]);
}

#[test]
fn deref_device_ptr() {
    let s = Series::new::<i32>("d", &[1, 2]).unwrap();
    let df = DataFrame::from_columns([("d", s)]).unwrap();

    let sv = df.series_view("d").unwrap();
    // device_data_ptr() is on ColumnView — accessed via Deref
    let ptr = sv.device_data_ptr().unwrap();
    assert!(!ptr.is_null());
}

// ---------------------------------------------------------------------------
// Series::view() returns SeriesView with correct name metadata
// ---------------------------------------------------------------------------

#[test]
fn unnamed_series_view() {
    let col = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let s = Series::from_column(col).unwrap();

    let sv = s.view().unwrap();
    assert_eq!(sv.name(), None);
    assert_eq!(sv.len().unwrap(), 3);
}

#[test]
fn named_series_view() {
    let s = Series::new::<i32>("foo", &[1, 2, 3]).unwrap();

    let sv = s.view().unwrap();
    assert_eq!(sv.name(), Some("foo"));
    assert_eq!(sv.data_type().unwrap().id(), TypeId::Int32);
}

// ---------------------------------------------------------------------------
// Debug output includes name, dtype, and len
// ---------------------------------------------------------------------------

#[test]
fn debug_shows_name_and_dtype() {
    let s = Series::new::<i32>("col", &[1, 2]).unwrap();
    let df = DataFrame::from_columns([("col", s)]).unwrap();

    let sv = df.series_view("col").unwrap();
    let dbg = format!("{:?}", sv);
    assert!(dbg.contains("col"), "Debug should contain name, got: {}", dbg);
    assert!(dbg.contains("Int32"), "Debug should contain dtype, got: {}", dbg);
}

// ---------------------------------------------------------------------------
// as_column_view() explicit accessor
// ---------------------------------------------------------------------------

#[test]
fn as_column_view_accessor() {
    let s = Series::new::<i32>("x", &[5, 6]).unwrap();
    let df = DataFrame::from_columns([("x", s)]).unwrap();

    let sv = df.series_view("x").unwrap();
    let cv = sv.as_column_view();
    assert_eq!(cv.len().unwrap(), 2);
}
