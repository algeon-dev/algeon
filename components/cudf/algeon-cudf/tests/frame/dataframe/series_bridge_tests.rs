//! Integration tests for Series-aware DataFrame constructors and accessors.
//!
//! All tests require a GPU.

use std::sync::Arc;

use cudf::{DataFrame, ErrorCode, ErrorKind, Series, TypeId, testing::prelude::*};

use crate::common::{assert_duplicate_name_error, assert_missing_column_error, error_context_value};

// ---------------------------------------------------------------------------
// from_series: happy path
// ---------------------------------------------------------------------------

fn test_from_series_happy_path_dynamic_iterator() {
    let ids = crate::common::series::<i32>("ids", &[1, 2, 3]).unwrap();
    let scores = crate::common::series::<f64>("scores", &[10.0, 20.0, 30.0]).unwrap();

    let df = DataFrame::from_series([("id", ids), ("score", scores)]).unwrap();

    assert_eq!(df.num_columns(), 2);
    assert_eq!(df.num_rows(), 3);
    assert_eq!(df.column_names(), &["id", "score"]);

    let col = df.series("id").unwrap();
    assert_eq!(col.to_vec_strict::<i32>().unwrap(), vec![1, 2, 3]);

    let col = df.series("score").unwrap();
    assert_eq!(col.to_vec_strict::<f64>().unwrap(), vec![10.0, 20.0, 30.0]);
}

fn test_from_series_single_column() {
    let vals = crate::common::series::<i64>("x", &[100, 200]).unwrap();
    let df = DataFrame::from_series([("x", vals)]).unwrap();

    assert_eq!(df.num_columns(), 1);
    assert_eq!(df.num_rows(), 2);
}

fn test_from_series_empty_series() {
    let empty = crate::common::series::<i32>("e", &[]).unwrap();
    let df = DataFrame::from_series([("e", empty)]).unwrap();

    assert_eq!(df.num_columns(), 1);
    assert_eq!(df.num_rows(), 0);
}

// ---------------------------------------------------------------------------
// from_series: rejects duplicates
// ---------------------------------------------------------------------------

fn test_from_series_rejects_duplicate_names() {
    let a = crate::common::series::<i32>("a", &[1, 2]).unwrap();
    let b = crate::common::series::<i32>("b", &[3, 4]).unwrap();

    let result = DataFrame::from_series([("x", a), ("x", b)]);

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert_duplicate_name_error(&err, "x");
}

// ---------------------------------------------------------------------------
// from_series: rejects length mismatch
// ---------------------------------------------------------------------------

fn test_from_series_rejects_length_mismatch_dynamic_iterator() {
    let a = crate::common::series::<i32>("a", &[1, 2, 3]).unwrap();
    let b = crate::common::series::<i32>("b", &[1, 2]).unwrap();

    let result = DataFrame::from_series([("x", a), ("y", b)]);

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert_eq!(err.code(), ErrorCode::ColumnLengthMismatch);
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(error_context_value(&err, "column_name").as_deref(), Some("y"));
    assert_eq!(error_context_value(&err, "actual_rows").as_deref(), Some("2"));
    assert_eq!(error_context_value(&err, "expected_rows").as_deref(), Some("3"));
}

// ---------------------------------------------------------------------------
// from_series: happy path (dynamic iterator)
// ---------------------------------------------------------------------------

fn test_from_series_happy_path() {
    let ids = crate::common::series::<i32>("ids", &[1, 2, 3]).unwrap();
    let names = crate::common::series("names", &["a", "b", "c"]).unwrap();

    let pairs = vec![("id".to_string(), ids), ("name".to_string(), names)];
    let df = DataFrame::from_series(pairs).unwrap();

    assert_eq!(df.num_columns(), 2);
    assert_eq!(df.num_rows(), 3);
    assert_eq!(df.column_names(), &["id", "name"]);
}

fn test_from_series_with_str_keys() {
    let a = crate::common::series::<i32>("a", &[10, 20]).unwrap();
    let b = crate::common::series::<f64>("b", &[1.0, 2.0]).unwrap();

    // Using &str keys (not String) to test Into<String> flexibility.
    let pairs = vec![("alpha", a), ("beta", b)];
    let df = DataFrame::from_series(pairs).unwrap();

    assert_eq!(df.column_names(), &["alpha", "beta"]);
}

fn test_from_series_empty_iterator() {
    let err = DataFrame::from_series(std::iter::empty::<(&str, Series)>()).unwrap_err();
    assert_eq!(err.code(), ErrorCode::EmptyConstructionRequiresRuntime);
    assert_eq!(err.kind(), ErrorKind::ExecutionContext);
    assert_eq!(err.operation(), "dataframe::from_series");
    assert!(err.message().contains("requires an explicit runtime"), "unexpected error: {err}");
}

fn test_from_series_rejects_duplicates() {
    let a = crate::common::series::<i32>("a", &[1]).unwrap();
    let b = crate::common::series::<i32>("b", &[2]).unwrap();

    let result = DataFrame::from_series(vec![("dup", a), ("dup", b)]);

    assert!(result.is_err());
    assert_eq!(result.unwrap_err().kind(), ErrorKind::InvalidInput);
}

fn test_from_series_rejects_length_mismatch() {
    let a = crate::common::series::<i32>("a", &[1, 2, 3]).unwrap();
    let b = crate::common::series::<i32>("b", &[1]).unwrap();

    let result = DataFrame::from_series(vec![("x", a), ("y", b)]);

    assert!(result.is_err());
    assert_eq!(result.unwrap_err().kind(), ErrorKind::InvalidInput);
}

// ---------------------------------------------------------------------------
// series(): returns sharing Arc
// ---------------------------------------------------------------------------

fn test_series_returns_series_values() {
    let ids = crate::common::series::<i32>("ids", &[1, 2, 3]).unwrap();
    let df = DataFrame::from_series([("id", ids)]).unwrap();

    let s = df.series("id").unwrap();
    assert_eq!(s.dtype().unwrap().id(), TypeId::Int32);
    assert_eq!(s.len(), 3);
}

fn test_series_shares_arc_with_dataframe() {
    let ids = crate::common::series::<i32>("ids", &[10, 20, 30]).unwrap();
    let df = DataFrame::from_series([("id", ids)]).unwrap();

    let s = df.series("id").unwrap();

    // The returned Series should share the same underlying GPU data
    // (Arc pointer equality).
    let df_col_ptr = df.series("id").unwrap().testing_view().unwrap().device_data_ptr().unwrap();
    let series_ptr = s.testing_view().unwrap().device_data_ptr().unwrap();
    assert_eq!(df_col_ptr, series_ptr);
}

fn test_series_nonexistent_column_returns_error() {
    let a = crate::common::series::<i32>("a", &[1]).unwrap();
    let df = DataFrame::from_series([("x", a)]).unwrap();

    let result = df.series("nonexistent");
    assert!(result.is_err());
    assert_missing_column_error(&result.unwrap_err(), "nonexistent");
}

// ---------------------------------------------------------------------------
// Round-trip: construct from Series, retrieve via series()
// ---------------------------------------------------------------------------

fn test_round_trip_from_explicit_columns_to_series() {
    let original = crate::common::series::<f64>("orig", &[1.0, 2.0, 3.0]).unwrap();
    let original_ptr = original.testing_view().unwrap().device_data_ptr().unwrap();

    let df = DataFrame::from_series([("val", original)]).unwrap();
    let retrieved = df.series("val").unwrap();
    assert_eq!(retrieved.len(), 3);
    assert_eq!(retrieved.dtype().unwrap().id(), TypeId::Float64);

    // The retrieved Series should share the same GPU allocation.
    let retrieved_ptr = retrieved.testing_view().unwrap().device_data_ptr().unwrap();
    assert_eq!(original_ptr, retrieved_ptr);
}

fn test_round_trip_from_series_to_series() {
    let a = crate::common::series::<i32>("a", &[1, 2]).unwrap();
    let b = crate::common::series::<i64>("b", &[100, 200]).unwrap();
    let a_ptr = a.testing_view().unwrap().device_data_ptr().unwrap();
    let b_ptr = b.testing_view().unwrap().device_data_ptr().unwrap();

    let df = DataFrame::from_series(vec![("alpha", a), ("beta", b)]).unwrap();

    let sa = df.series("alpha").unwrap();
    let sb = df.series("beta").unwrap();

    // Names come from the DataFrame keys, not the original Series names.

    // GPU data is shared (zero-copy).
    assert_eq!(sa.testing_view().unwrap().device_data_ptr().unwrap(), a_ptr);
    assert_eq!(sb.testing_view().unwrap().device_data_ptr().unwrap(), b_ptr);
}

// ---------------------------------------------------------------------------
// from_series preserves Arc sharing from input Series
// ---------------------------------------------------------------------------

fn test_from_series_preserves_arc_sharing() {
    let s = crate::common::series::<i32>("x", &[1, 2, 3]).unwrap();
    let arc_before = s.clone().into_inner();

    let df = DataFrame::from_series([("x", s)]).unwrap();
    let retrieved = df.series("x").unwrap();

    // The Arc inside the DataFrame should be the same Arc as the input Series.
    assert!(Arc::ptr_eq(&arc_before, &retrieved.into_inner()));
}

#[test]
fn series_bridge_tests_cases() {
    crate::common::run_named_cases(&[
        ("test_from_series_happy_path_dynamic_iterator", test_from_series_happy_path_dynamic_iterator),
        ("test_from_series_single_column", test_from_series_single_column),
        ("test_from_series_empty_series", test_from_series_empty_series),
        ("test_from_series_rejects_duplicate_names", test_from_series_rejects_duplicate_names),
        (
            "test_from_series_rejects_length_mismatch_dynamic_iterator",
            test_from_series_rejects_length_mismatch_dynamic_iterator,
        ),
        ("test_from_series_happy_path", test_from_series_happy_path),
        ("test_from_series_with_str_keys", test_from_series_with_str_keys),
        ("test_from_series_empty_iterator", test_from_series_empty_iterator),
        ("test_from_series_rejects_duplicates", test_from_series_rejects_duplicates),
        ("test_from_series_rejects_length_mismatch", test_from_series_rejects_length_mismatch),
        ("test_series_returns_series_values", test_series_returns_series_values),
        ("test_series_shares_arc_with_dataframe", test_series_shares_arc_with_dataframe),
        ("test_series_nonexistent_column_returns_error", test_series_nonexistent_column_returns_error),
        ("test_round_trip_from_explicit_columns_to_series", test_round_trip_from_explicit_columns_to_series),
        ("test_round_trip_from_series_to_series", test_round_trip_from_series_to_series),
        ("test_from_series_preserves_arc_sharing", test_from_series_preserves_arc_sharing),
    ]);
}
