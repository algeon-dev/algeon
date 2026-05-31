//! Tests for the `internal` module's infrastructure.
//!
//! `TempTableView`, `resolve`, and `reassemble` are `pub(crate)`, so we
//! verify them **indirectly** through public DataFrame operations that
//! exercise the internal helpers.
//!
//! Direct unit tests for the helpers live inside `src/internal/mod.rs`.

use cudf::{Column, DataFrame, Series};

// ---------------------------------------------------------------------------
// TempTableView — exercised via filter, sort, distinct, drop_nulls
// These operations internally build a TempTableView from column views.
// ---------------------------------------------------------------------------

#[test]
fn filter_preserves_row_and_column_counts() {
    // filter internally uses build_table_view / TempTableView infrastructure.
    let ids = Series::new::<i32>("ids", &[1, 2, 3, 4, 5]).unwrap();
    let vals = Series::new::<f64>("vals", &[10.0, 20.0, 30.0, 40.0, 50.0]).unwrap();
    let df = DataFrame::from_columns([("id", ids), ("val", vals)]).unwrap();

    // Boolean mask: keep rows 0, 2, 4.
    let mask = Column::from_slice(&[true, false, true, false, true]).unwrap();
    let mask_view = mask.view().unwrap();
    let filtered = df.filter(&mask_view).unwrap();

    assert_eq!(filtered.num_columns(), 2, "column count preserved");
    assert_eq!(filtered.num_rows().unwrap(), 3, "filtered row count");
    assert_eq!(filtered.column_names(), vec!["id", "val"], "schema names preserved");
}

#[test]
fn sort_preserves_schema_names_and_lengths() {
    // sort internally resolves column names and builds table views.
    let names_col = Series::new::<i32>("a", &[3, 1, 2]).unwrap();
    let vals_col = Series::new::<i64>("b", &[30, 10, 20]).unwrap();
    let df = DataFrame::from_columns([("a", names_col), ("b", vals_col)]).unwrap();

    let sorted = df.sort(&["a"], &[cudf::sorting::Order::Ascending]).unwrap();

    assert_eq!(sorted.num_columns(), 2);
    assert_eq!(sorted.num_rows().unwrap(), 3);
    assert_eq!(sorted.column_names(), vec!["a", "b"]);

    // Verify sorted order.
    let a_view = sorted.column("a").unwrap();
    let a_vals = a_view.to_vec::<i32>().unwrap();
    assert_eq!(a_vals, vec![1, 2, 3]);
}

#[test]
fn reassembly_after_distinct_preserves_schema() {
    // distinct internally uses resolve + reassemble helpers.
    let a = Series::new::<i32>("x", &[1, 1, 2, 2, 3]).unwrap();
    let b = Series::new::<i64>("y", &[10, 10, 20, 20, 30]).unwrap();
    let df = DataFrame::from_columns([("x", a), ("y", b)]).unwrap();

    let distinct = df.distinct(&["x", "y"], cudf::stream_compaction::DuplicateKeepOption::KeepFirst).unwrap();

    assert_eq!(distinct.num_columns(), 2, "schema columns preserved");
    assert_eq!(distinct.num_rows().unwrap(), 3, "distinct row count");
    assert_eq!(distinct.column_names(), vec!["x", "y"], "schema names preserved");
}

// ---------------------------------------------------------------------------
// resolve helpers — column name resolution errors
// ---------------------------------------------------------------------------

#[test]
fn sort_with_missing_column_returns_error() {
    // resolve_column_names should produce a clear error for unknown columns.
    let a = Series::new::<i32>("a", &[1, 2, 3]).unwrap();
    let df = DataFrame::from_columns([("a", a)]).unwrap();

    let err = df.sort(&["nonexistent"], &[cudf::sorting::Order::Ascending]).unwrap_err();
    assert_eq!(err.kind(), cudf::ErrorKind::InvalidOperation);
    let msg = err.to_string();
    assert!(msg.contains("nonexistent"), "error should name the missing column: {}", msg);
}

// ---------------------------------------------------------------------------
// reassemble helpers — from_table_with_names via join
// ---------------------------------------------------------------------------

#[test]
fn join_reassembly_preserves_schema_names() {
    // join internally uses reassembly helpers to rebuild DataFrames.
    let left_key = Series::new::<i32>("key", &[1, 2, 3]).unwrap();
    let left_val = Series::new::<f64>("lval", &[10.0, 20.0, 30.0]).unwrap();
    let left = DataFrame::from_columns([("key", left_key), ("lval", left_val)]).unwrap();

    let right_key = Series::new::<i32>("key", &[2, 3, 4]).unwrap();
    let right_val = Series::new::<f64>("rval", &[200.0, 300.0, 400.0]).unwrap();
    let right = DataFrame::from_columns([("key", right_key), ("rval", right_val)]).unwrap();

    let joined = left.inner_join(&right, &["key"]).unwrap();

    // Inner join on [2,3] → 2 rows.
    assert_eq!(joined.num_rows().unwrap(), 2);
    // Schema: key + lval + rval.
    assert_eq!(joined.column_names(), vec!["key", "lval", "rval"]);
}

// ---------------------------------------------------------------------------
// validate_no_duplicate_names — exercised via DataFrame constructors
// ---------------------------------------------------------------------------

#[test]
fn from_series_rejects_duplicate_names() {
    let a = Series::new::<i32>("a", &[1]).unwrap();
    let b = Series::new::<i32>("b", &[2]).unwrap();

    let err = DataFrame::from_series(vec![("x".to_string(), a), ("x".to_string(), b)]).unwrap_err();
    assert_eq!(err.kind(), cudf::ErrorKind::InvalidOperation);
    let msg = err.to_string();
    assert!(msg.contains("duplicate"), "error message should mention duplicate: {}", msg);
}

#[test]
fn dataframe_new_rejects_duplicate_names() {
    let c1 = Column::from_slice(&[1i32]).unwrap();
    let c2 = Column::from_slice(&[2i32]).unwrap();
    let err = DataFrame::new(vec!["dup".into(), "dup".into()], vec![c1, c2]).unwrap_err();
    assert_eq!(err.kind(), cudf::ErrorKind::InvalidOperation);
    let msg = err.to_string();
    assert!(msg.contains("duplicate"), "error message should mention duplicate: {}", msg);
}
