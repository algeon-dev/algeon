//! Integration tests for Series-aware DataFrame constructors and accessors.
//!
//! All tests require a GPU.

use std::sync::Arc;

use cudf::{DataFrame, ErrorKind, Series, TypeId};

use crate::common::{assert_duplicate_name_error, assert_missing_column_error, error_context_value};

// ---------------------------------------------------------------------------
// from_columns: happy path
// ---------------------------------------------------------------------------

#[test]
fn test_from_columns_happy_path() {
    let ids = Series::new::<i32>("ids", &[1, 2, 3]).unwrap();
    let scores = Series::new::<f64>("scores", &[10.0, 20.0, 30.0]).unwrap();

    let df = DataFrame::from_named_pairs([("id", ids), ("score", scores)]).unwrap();

    assert_eq!(df.num_columns(), 2);
    assert_eq!(df.num_rows(), 3);
    assert_eq!(df.column_names(), &["id", "score"]);

    let col = df.series("id").unwrap();
    assert_eq!(col.to_vec_strict::<i32>().unwrap(), vec![1, 2, 3]);

    let col = df.series("score").unwrap();
    assert_eq!(col.to_vec_strict::<f64>().unwrap(), vec![10.0, 20.0, 30.0]);
}

#[test]
fn test_from_columns_single_column() {
    let vals = Series::new::<i64>("x", &[100, 200]).unwrap();
    let df = DataFrame::from_series([vals]).unwrap();

    assert_eq!(df.num_columns(), 1);
    assert_eq!(df.num_rows(), 2);
}

#[test]
fn test_from_columns_empty_series() {
    let empty = Series::new::<i32>("e", &[]).unwrap();
    let df = DataFrame::from_series([empty]).unwrap();

    assert_eq!(df.num_columns(), 1);
    assert_eq!(df.num_rows(), 0);
}

// ---------------------------------------------------------------------------
// from_columns: rejects duplicates
// ---------------------------------------------------------------------------

#[test]
fn test_from_columns_rejects_duplicate_names() {
    let a = Series::new::<i32>("a", &[1, 2]).unwrap();
    let b = Series::new::<i32>("b", &[3, 4]).unwrap();

    let result = DataFrame::from_named_pairs([("x", a), ("x", b)]);

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert_duplicate_name_error(&err, "x");
}

// ---------------------------------------------------------------------------
// from_columns: rejects length mismatch
// ---------------------------------------------------------------------------

#[test]
fn test_from_columns_rejects_length_mismatch() {
    let a = Series::new::<i32>("a", &[1, 2, 3]).unwrap();
    let b = Series::new::<i32>("b", &[1, 2]).unwrap();

    let result = DataFrame::from_named_pairs([("x", a), ("y", b)]);

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
    assert_eq!(error_context_value(&err, "column_name"), Some("y"));
    assert_eq!(error_context_value(&err, "column_rows"), Some("2"));
    assert_eq!(error_context_value(&err, "expected_rows"), Some("3"));
}

// ---------------------------------------------------------------------------
// from_named_pairs: happy path (dynamic iterator)
// ---------------------------------------------------------------------------

#[test]
fn test_from_named_pairs_happy_path() {
    let ids = Series::new::<i32>("ids", &[1, 2, 3]).unwrap();
    let names = Series::new("names", &["a", "b", "c"]).unwrap();

    let pairs = vec![("id".to_string(), ids), ("name".to_string(), names)];
    let df = DataFrame::from_named_pairs(pairs).unwrap();

    assert_eq!(df.num_columns(), 2);
    assert_eq!(df.num_rows(), 3);
    assert_eq!(df.column_names(), &["id", "name"]);
}

#[test]
fn test_from_named_pairs_with_str_keys() {
    let a = Series::new::<i32>("a", &[10, 20]).unwrap();
    let b = Series::new::<f64>("b", &[1.0, 2.0]).unwrap();

    // Using &str keys (not String) to test Into<String> flexibility.
    let pairs = vec![("alpha", a), ("beta", b)];
    let df = DataFrame::from_named_pairs(pairs).unwrap();

    assert_eq!(df.column_names(), &["alpha", "beta"]);
}

#[test]
fn test_from_named_pairs_empty_iterator() {
    let df = DataFrame::from_named_pairs(std::iter::empty::<(&str, Series)>()).unwrap();

    assert_eq!(df.num_columns(), 0);
    assert_eq!(df.num_rows(), 0);
}

#[test]
fn test_from_named_pairs_rejects_duplicates() {
    let a = Series::new::<i32>("a", &[1]).unwrap();
    let b = Series::new::<i32>("b", &[2]).unwrap();

    let result = DataFrame::from_named_pairs(vec![("dup", a), ("dup", b)]);

    assert!(result.is_err());
    assert_eq!(result.unwrap_err().kind(), ErrorKind::InvalidOperation);
}

#[test]
fn test_from_named_pairs_rejects_length_mismatch() {
    let a = Series::new::<i32>("a", &[1, 2, 3]).unwrap();
    let b = Series::new::<i32>("b", &[1]).unwrap();

    let result = DataFrame::from_named_pairs(vec![("x", a), ("y", b)]);

    assert!(result.is_err());
    assert_eq!(result.unwrap_err().kind(), ErrorKind::InvalidOperation);
}

// ---------------------------------------------------------------------------
// series(): returns sharing Arc
// ---------------------------------------------------------------------------

#[test]
fn test_series_returns_named_series() {
    let ids = Series::new::<i32>("ids", &[1, 2, 3]).unwrap();
    let df = DataFrame::from_named_pairs([("id", ids)]).unwrap();

    let s = df.series("id").unwrap();

    assert_eq!(s.name(), Some("id"));
    assert_eq!(s.dtype().unwrap().id(), TypeId::Int32);
    assert_eq!(s.len(), 3);
}

#[test]
fn test_series_shares_arc_with_dataframe() {
    let ids = Series::new::<i32>("ids", &[10, 20, 30]).unwrap();
    let df = DataFrame::from_named_pairs([("id", ids)]).unwrap();

    let s = df.series("id").unwrap();

    // The returned Series should share the same underlying GPU data
    // (Arc pointer equality).
    let df_col_ptr = df.series("id").unwrap().as_column_view().unwrap().device_data_ptr().unwrap();
    let series_ptr = s.as_column_view().unwrap().device_data_ptr().unwrap();
    assert_eq!(df_col_ptr, series_ptr);
}

#[test]
fn test_series_nonexistent_column_returns_error() {
    let a = Series::new::<i32>("a", &[1]).unwrap();
    let df = DataFrame::from_named_pairs([("x", a)]).unwrap();

    let result = df.series("nonexistent");
    assert!(result.is_err());
    assert_missing_column_error(&result.unwrap_err(), "nonexistent");
}

// ---------------------------------------------------------------------------
// Round-trip: construct from Series, retrieve via series()
// ---------------------------------------------------------------------------

#[test]
fn test_round_trip_from_columns_to_series() {
    let original = Series::new::<f64>("orig", &[1.0, 2.0, 3.0]).unwrap();
    let original_ptr = original.as_column_view().unwrap().device_data_ptr().unwrap();

    let df = DataFrame::from_named_pairs([("val", original)]).unwrap();
    let retrieved = df.series("val").unwrap();

    assert_eq!(retrieved.name(), Some("val"));
    assert_eq!(retrieved.len(), 3);
    assert_eq!(retrieved.dtype().unwrap().id(), TypeId::Float64);

    // The retrieved Series should share the same GPU allocation.
    let retrieved_ptr = retrieved.as_column_view().unwrap().device_data_ptr().unwrap();
    assert_eq!(original_ptr, retrieved_ptr);
}

#[test]
fn test_round_trip_from_named_pairs_to_series() {
    let a = Series::new::<i32>("a", &[1, 2]).unwrap();
    let b = Series::new::<i64>("b", &[100, 200]).unwrap();
    let a_ptr = a.as_column_view().unwrap().device_data_ptr().unwrap();
    let b_ptr = b.as_column_view().unwrap().device_data_ptr().unwrap();

    let df = DataFrame::from_named_pairs(vec![("alpha", a), ("beta", b)]).unwrap();

    let sa = df.series("alpha").unwrap();
    let sb = df.series("beta").unwrap();

    // Names come from the DataFrame keys, not the original Series names.
    assert_eq!(sa.name(), Some("alpha"));
    assert_eq!(sb.name(), Some("beta"));

    // GPU data is shared (zero-copy).
    assert_eq!(sa.as_column_view().unwrap().device_data_ptr().unwrap(), a_ptr);
    assert_eq!(sb.as_column_view().unwrap().device_data_ptr().unwrap(), b_ptr);
}

// ---------------------------------------------------------------------------
// from_columns preserves Arc sharing from input Series
// ---------------------------------------------------------------------------

#[test]
fn test_from_columns_preserves_arc_sharing() {
    let s = Series::new::<i32>("x", &[1, 2, 3]).unwrap();
    let arc_before = s.clone().into_inner();

    let df = DataFrame::from_series([s]).unwrap();
    let retrieved = df.series("x").unwrap();

    // The Arc inside the DataFrame should be the same Arc as the input Series.
    assert!(Arc::ptr_eq(&arc_before, &retrieved.into_inner()));
}
