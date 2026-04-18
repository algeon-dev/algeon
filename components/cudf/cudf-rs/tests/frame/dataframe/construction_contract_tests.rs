//! Tests for DataFrame construction contracts.
//!
//! These cases protect how schema names are sourced during construction:
//! `DataFrame::from_series` uses `Series` names, while
//! `DataFrame::from_named_pairs` applies explicit schema names.
use cudf::testing::prelude::*;


use cudf::{DataFrame, ErrorKind, Runtime, Series};

use crate::common::column_from_slice;

// ---------------------------------------------------------------------------
// from_series: Series-carried names
// ---------------------------------------------------------------------------

#[test]
fn test_from_series_uses_series_names() {
    let x = crate::common::series::<i32>("x", &[1, 2, 3]).unwrap();
    let y = crate::common::series::<f64>("y", &[10.0, 20.0, 30.0]).unwrap();

    let df = DataFrame::from_series([x, y]).unwrap();

    assert_eq!(df.num_columns(), 2);
    assert_eq!(df.num_rows(), 3);
    assert_eq!(df.column_names(), vec!["x", "y"]);
}

#[test]
fn test_from_series_rejects_unnamed() {
    let named = crate::common::series::<i32>("a", &[1, 2]).unwrap();
    let unnamed = Series::from_column(column_from_slice(&[3i32, 4]).unwrap()).unwrap();

    let err = DataFrame::from_series([named, unnamed]).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
}

#[test]
fn test_from_series_rejects_duplicate_names() {
    let a1 = crate::common::series::<i32>("dup", &[1, 2]).unwrap();
    let a2 = crate::common::series::<i32>("dup", &[3, 4]).unwrap();

    let err = DataFrame::from_series([a1, a2]).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
}

#[test]
fn test_from_series_empty_iterator() {
    let err = DataFrame::from_series(Vec::<Series>::new()).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
    assert_eq!(err.operation(), "dataframe::from_series");
    assert!(err.message().contains("requires an explicit runtime"), "unexpected error: {err}");
}

#[test]
fn test_from_named_pairs_empty_iterator_requires_runtime() {
    let err = DataFrame::from_named_pairs(Vec::<(&str, Series)>::new()).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
    assert_eq!(err.operation(), "dataframe::from_named_pairs");
    assert!(err.message().contains("requires an explicit runtime"), "unexpected error: {err}");
}

#[test]
fn test_runtime_empty_dataframe_creates_explicit_zero_by_zero_frame() {
    let rt = Runtime::new().unwrap();
    let df = rt.empty_dataframe();
    assert_eq!(df.num_columns(), 0);
    assert_eq!(df.num_rows(), 0);
    assert_eq!(df.testing_execution_domain_kind(), "explicit");
    assert_eq!(df.testing_execution_domain_id(), rt.testing_execution_domain_id());
}

#[test]
fn test_from_series_mismatched_lengths() {
    let a = crate::common::series::<i32>("a", &[1, 2, 3]).unwrap();
    let b = crate::common::series::<i32>("b", &[1, 2]).unwrap();

    let err = DataFrame::from_series([a, b]).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
}

// ---------------------------------------------------------------------------
// from_named_pairs: explicit schema override
// ---------------------------------------------------------------------------

#[test]
fn test_from_named_pairs_basic() {
    let x = crate::common::series::<i32>("source_x", &[1, 2, 3]).unwrap();
    let y = crate::common::series::<f64>("source_y", &[10.0, 20.0, 30.0]).unwrap();

    // Schema key overrides the Series' own name.
    let df = DataFrame::from_named_pairs([("x", x), ("y", y)]).unwrap();

    assert_eq!(df.num_columns(), 2);
    assert_eq!(df.column_names(), vec!["x", "y"]);
}

#[test]
fn test_from_named_pairs_rejects_duplicates() {
    let a = crate::common::series::<i32>("a", &[1]).unwrap();
    let b = crate::common::series::<i32>("b", &[2]).unwrap();

    let err = DataFrame::from_named_pairs([("same", a), ("same", b)]).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
}

// ---------------------------------------------------------------------------
// dataframe! macro
// ---------------------------------------------------------------------------

#[test]
fn test_dataframe_macro_uses_from_named_pairs() {
    let x = crate::common::series::<i32>("src", &[1, 2]).unwrap();
    let y = crate::common::series::<f64>("src", &[3.0, 4.0]).unwrap();

    let df = cudf::dataframe! { "a" => x, "b" => y }.unwrap();
    assert_eq!(df.column_names(), vec!["a", "b"]);
}
