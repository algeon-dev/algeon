#![allow(deprecated)] // Tests exercise deprecated DataFrame::column()
//! Integration tests for `DataFrameView` — a named borrow of all DataFrame columns.
//!
//! All tests require a GPU.

use cudf::{DataFrame, series::Series};

// ---------------------------------------------------------------------------
// Helper: build a small DataFrame with known data
// ---------------------------------------------------------------------------

fn sample_df() -> DataFrame {
    let a = Series::new::<i32>("a", &[1, 2, 3]).unwrap();
    let b = Series::new::<f64>("b", &[10.0, 20.0, 30.0]).unwrap();
    let c = Series::new::<i64>("c", &[100, 200, 300]).unwrap();
    DataFrame::from_columns([("a", a), ("b", b), ("c", c)]).unwrap()
}

// ---------------------------------------------------------------------------
// view_column_names — schema order preserved
// ---------------------------------------------------------------------------

#[test]
fn view_column_names() {
    let df = sample_df();
    let view = df.view();
    assert_eq!(view.column_names(), vec!["a", "b", "c"]);
}

// ---------------------------------------------------------------------------
// view_column_by_name — returns Some(SeriesView) with correct name + data
// ---------------------------------------------------------------------------

#[test]
fn view_column_by_name() {
    let df = sample_df();
    let view = df.view();

    let sv = view.column("b").expect("column 'b' should exist");
    assert_eq!(sv.name(), Some("b"));
    let vals: Vec<f64> = sv.to_vec().unwrap();
    assert_eq!(vals, vec![10.0, 20.0, 30.0]);
}

// ---------------------------------------------------------------------------
// view_column_miss — returns None
// ---------------------------------------------------------------------------

#[test]
fn view_column_miss() {
    let df = sample_df();
    let view = df.view();
    assert!(view.column("z").is_none());
}

// ---------------------------------------------------------------------------
// view_num_columns — matches schema
// ---------------------------------------------------------------------------

#[test]
fn view_num_columns() {
    let df = sample_df();
    let view = df.view();
    assert_eq!(view.num_columns(), 3);
}

// ---------------------------------------------------------------------------
// view_iter — correct count and names
// ---------------------------------------------------------------------------

#[test]
fn view_iter() {
    let df = sample_df();
    let view = df.view();

    let names: Vec<&str> = view.iter().map(|(name, _cv)| name).collect();
    assert_eq!(names, vec!["a", "b", "c"]);
    assert_eq!(view.iter().count(), 3);
}

// ---------------------------------------------------------------------------
// view_zero_copy — device pointers match df.column() pointers
// ---------------------------------------------------------------------------

#[test]
fn view_zero_copy() {
    let df = sample_df();
    let view = df.view();

    for name in ["a", "b", "c"] {
        let ptr_from_df = df.column(name).unwrap().device_data_ptr().unwrap();
        let ptr_from_view = view.column(name).unwrap().device_data_ptr().unwrap();
        assert_eq!(ptr_from_df, ptr_from_view, "device pointers should match for column '{}'", name);
    }
}
