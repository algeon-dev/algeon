//! Tests for DataFrame construction contracts.
//!
//! These cases protect the rule that `DataFrame` is the schema authority:
//! constructors take explicit output names, and `Series` values are nameless.
use cudf::{DataFrame, ErrorCode, ErrorKind, Runtime, Series, testing::prelude::*};

use crate::common::{assert_duplicate_name_error, column_from_slice, error_context_value};

fn assert_length_mismatch_contract(
    err: &cudf::Error,
    operation: &str,
    expected_name: &str,
    actual_rows: &str,
    expected_rows: &str,
) {
    assert_eq!(err.code(), ErrorCode::ColumnLengthMismatch);
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.operation(), operation);
    assert_eq!(error_context_value(err, "column_name").as_deref(), Some(expected_name));
    assert_eq!(error_context_value(err, "actual_rows").as_deref(), Some(actual_rows));
    assert_eq!(error_context_value(err, "expected_rows").as_deref(), Some(expected_rows));
}

// ---------------------------------------------------------------------------
// from_series: explicit schema names
// ---------------------------------------------------------------------------

#[test]
fn test_from_series_uses_explicit_names() {
    let x = crate::common::series::<i32>("source_x", &[1, 2, 3]).unwrap();
    let y = crate::common::series::<f64>("source_y", &[10.0, 20.0, 30.0]).unwrap();

    let df = DataFrame::from_series([("x", x), ("y", y)]).unwrap();

    assert_eq!(df.num_columns(), 2);
    assert_eq!(df.num_rows(), 3);
    assert_eq!(df.column_names(), vec!["x", "y"]);
}

#[test]
fn test_from_series_accepts_unnamed_series() {
    let unnamed = Series::from_column(column_from_slice(&[3i32, 4]).unwrap()).unwrap();
    let named = crate::common::series::<i32>("a", &[1, 2]).unwrap();

    let df = DataFrame::from_series([("left", named), ("right", unnamed)]).unwrap();
    assert_eq!(df.column_names(), vec!["left", "right"]);
}

#[test]
fn test_from_series_rejects_duplicate_names() {
    let a1 = crate::common::series::<i32>("left", &[1, 2]).unwrap();
    let a2 = crate::common::series::<i32>("right", &[3, 4]).unwrap();

    let err = DataFrame::from_series([("dup", a1), ("dup", a2)]).unwrap_err();
    assert_duplicate_name_error(&err, "dup");
    assert_eq!(error_context_value(&err, "column_name_count").as_deref(), Some("2"));
}

#[test]
fn test_from_series_empty_iterator() {
    let err = DataFrame::from_series(Vec::<(&str, Series)>::new()).unwrap_err();
    assert_eq!(err.code(), ErrorCode::EmptyConstructionRequiresRuntime);
    assert_eq!(err.kind(), ErrorKind::ExecutionContext);
    assert_eq!(err.operation(), "dataframe::from_series");
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

    let err = DataFrame::from_series([("x", a), ("y", b)]).unwrap_err();
    assert_length_mismatch_contract(&err, "dataframe::from_series", "y", "2", "3");
}

#[test]
fn test_dataframe_series_accessor_uses_schema_name_not_original_series_name() {
    let source = crate::common::series::<i32>("source_name", &[7, 8, 9]).unwrap();

    let df = DataFrame::from_series([("schema_name", source)]).unwrap();
    let projected = df.series("schema_name").unwrap();
    assert_eq!(projected.to_vec_strict::<i32>().unwrap(), vec![7, 8, 9]);
    assert_eq!(df.column_names(), vec!["schema_name"]);
}

// ---------------------------------------------------------------------------
// dataframe! macro
// ---------------------------------------------------------------------------

#[test]
fn test_dataframe_macro_uses_from_series() {
    let x = crate::common::series::<i32>("src", &[1, 2]).unwrap();
    let y = crate::common::series::<f64>("src", &[3.0, 4.0]).unwrap();

    let df = cudf::dataframe! { "a" => x, "b" => y }.unwrap();
    assert_eq!(df.column_names(), vec!["a", "b"]);
}
