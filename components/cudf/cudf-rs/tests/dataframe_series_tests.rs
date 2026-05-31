//! Integration tests for Series-aware DataFrame constructors and accessors.
//!
//! All tests require a GPU.

use std::sync::Arc;

use cudf::{dataframe::DataFrame, error::ErrorKind, series::Series, types::TypeId};

// ---------------------------------------------------------------------------
// from_columns: happy path
// ---------------------------------------------------------------------------

#[test]
fn test_from_columns_happy_path() {
    let ids = Series::new::<i32>("ids", &[1, 2, 3]).unwrap();
    let scores = Series::new::<f64>("scores", &[10.0, 20.0, 30.0]).unwrap();

    let df = DataFrame::from_columns([("id", ids), ("score", scores)]).unwrap();

    assert_eq!(df.num_columns(), 2);
    assert_eq!(df.num_rows().unwrap(), 3);
    assert_eq!(df.column_names(), &["id", "score"]);

    let col = df.column("id").unwrap();
    assert_eq!(col.to_vec::<i32>().unwrap(), vec![1, 2, 3]);

    let col = df.column("score").unwrap();
    assert_eq!(col.to_vec::<f64>().unwrap(), vec![10.0, 20.0, 30.0]);
}

#[test]
fn test_from_columns_single_column() {
    let vals = Series::new::<i64>("x", &[100, 200]).unwrap();
    let df = DataFrame::from_columns([("x", vals)]).unwrap();

    assert_eq!(df.num_columns(), 1);
    assert_eq!(df.num_rows().unwrap(), 2);
}

#[test]
fn test_from_columns_empty_series() {
    let empty = Series::new::<i32>("e", &[]).unwrap();
    let df = DataFrame::from_columns([("e", empty)]).unwrap();

    assert_eq!(df.num_columns(), 1);
    assert_eq!(df.num_rows().unwrap(), 0);
}

// ---------------------------------------------------------------------------
// from_columns: rejects duplicates
// ---------------------------------------------------------------------------

#[test]
fn test_from_columns_rejects_duplicate_names() {
    let a = Series::new::<i32>("a", &[1, 2]).unwrap();
    let b = Series::new::<i32>("b", &[3, 4]).unwrap();

    let result = DataFrame::from_columns([("x", a), ("x", b)]);

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
    assert!(err.to_string().contains("duplicate"));
}

// ---------------------------------------------------------------------------
// from_columns: rejects length mismatch
// ---------------------------------------------------------------------------

#[test]
fn test_from_columns_rejects_length_mismatch() {
    let a = Series::new::<i32>("a", &[1, 2, 3]).unwrap();
    let b = Series::new::<i32>("b", &[1, 2]).unwrap();

    let result = DataFrame::from_columns([("x", a), ("y", b)]);

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
    assert!(err.to_string().contains("rows"));
}

// ---------------------------------------------------------------------------
// from_series: happy path (dynamic iterator)
// ---------------------------------------------------------------------------

#[test]
fn test_from_series_happy_path() {
    let ids = Series::new::<i32>("ids", &[1, 2, 3]).unwrap();
    let names = Series::from_strings("names", &["a", "b", "c"]).unwrap();

    let pairs = vec![("id".to_string(), ids), ("name".to_string(), names)];
    let df = DataFrame::from_series(pairs).unwrap();

    assert_eq!(df.num_columns(), 2);
    assert_eq!(df.num_rows().unwrap(), 3);
    assert_eq!(df.column_names(), &["id", "name"]);
}

#[test]
fn test_from_series_with_str_keys() {
    let a = Series::new::<i32>("a", &[10, 20]).unwrap();
    let b = Series::new::<f64>("b", &[1.0, 2.0]).unwrap();

    // Using &str keys (not String) to test Into<String> flexibility.
    let pairs = vec![("alpha", a), ("beta", b)];
    let df = DataFrame::from_series(pairs).unwrap();

    assert_eq!(df.column_names(), &["alpha", "beta"]);
}

#[test]
fn test_from_series_empty_iterator() {
    let df = DataFrame::from_series(std::iter::empty::<(&str, Series)>()).unwrap();

    assert_eq!(df.num_columns(), 0);
    assert_eq!(df.num_rows().unwrap(), 0);
}

#[test]
fn test_from_series_rejects_duplicates() {
    let a = Series::new::<i32>("a", &[1]).unwrap();
    let b = Series::new::<i32>("b", &[2]).unwrap();

    let result = DataFrame::from_series(vec![("dup", a), ("dup", b)]);

    assert!(result.is_err());
    assert_eq!(result.unwrap_err().kind(), ErrorKind::InvalidOperation);
}

#[test]
fn test_from_series_rejects_length_mismatch() {
    let a = Series::new::<i32>("a", &[1, 2, 3]).unwrap();
    let b = Series::new::<i32>("b", &[1]).unwrap();

    let result = DataFrame::from_series(vec![("x", a), ("y", b)]);

    assert!(result.is_err());
    assert_eq!(result.unwrap_err().kind(), ErrorKind::InvalidOperation);
}

// ---------------------------------------------------------------------------
// series(): returns sharing Arc
// ---------------------------------------------------------------------------

#[test]
fn test_series_returns_named_series() {
    let ids = Series::new::<i32>("ids", &[1, 2, 3]).unwrap();
    let df = DataFrame::from_columns([("id", ids)]).unwrap();

    let s = df.series("id").unwrap();

    assert_eq!(s.name(), Some("id"));
    assert_eq!(s.dtype().unwrap().id(), TypeId::Int32);
    assert_eq!(s.len().unwrap(), 3);
}

#[test]
fn test_series_shares_arc_with_dataframe() {
    let ids = Series::new::<i32>("ids", &[10, 20, 30]).unwrap();
    let df = DataFrame::from_columns([("id", ids)]).unwrap();

    let s = df.series("id").unwrap();

    // The returned Series should share the same underlying GPU data
    // (Arc pointer equality).
    let df_col_ptr = df.column("id").unwrap().device_data_ptr().unwrap();
    let series_ptr = s.view().unwrap().device_data_ptr().unwrap();
    assert_eq!(df_col_ptr, series_ptr);
}

#[test]
fn test_series_nonexistent_column_returns_error() {
    let a = Series::new::<i32>("a", &[1]).unwrap();
    let df = DataFrame::from_columns([("x", a)]).unwrap();

    let result = df.series("nonexistent");
    assert!(result.is_err());
    assert_eq!(result.unwrap_err().kind(), ErrorKind::InvalidOperation);
}

// ---------------------------------------------------------------------------
// Round-trip: construct from Series, retrieve via series()
// ---------------------------------------------------------------------------

#[test]
fn test_round_trip_from_columns_to_series() {
    let original = Series::new::<f64>("orig", &[1.0, 2.0, 3.0]).unwrap();
    let original_ptr = original.view().unwrap().device_data_ptr().unwrap();

    let df = DataFrame::from_columns([("val", original)]).unwrap();
    let retrieved = df.series("val").unwrap();

    assert_eq!(retrieved.name(), Some("val"));
    assert_eq!(retrieved.len().unwrap(), 3);
    assert_eq!(retrieved.dtype().unwrap().id(), TypeId::Float64);

    // The retrieved Series should share the same GPU allocation.
    let retrieved_ptr = retrieved.view().unwrap().device_data_ptr().unwrap();
    assert_eq!(original_ptr, retrieved_ptr);
}

#[test]
fn test_round_trip_from_series_to_series() {
    let a = Series::new::<i32>("a", &[1, 2]).unwrap();
    let b = Series::new::<i64>("b", &[100, 200]).unwrap();
    let a_ptr = a.view().unwrap().device_data_ptr().unwrap();
    let b_ptr = b.view().unwrap().device_data_ptr().unwrap();

    let df = DataFrame::from_series(vec![("alpha", a), ("beta", b)]).unwrap();

    let sa = df.series("alpha").unwrap();
    let sb = df.series("beta").unwrap();

    // Names come from the DataFrame keys, not the original Series names.
    assert_eq!(sa.name(), Some("alpha"));
    assert_eq!(sb.name(), Some("beta"));

    // GPU data is shared (zero-copy).
    assert_eq!(sa.view().unwrap().device_data_ptr().unwrap(), a_ptr);
    assert_eq!(sb.view().unwrap().device_data_ptr().unwrap(), b_ptr);
}

// ---------------------------------------------------------------------------
// from_columns preserves Arc sharing from input Series
// ---------------------------------------------------------------------------

#[test]
fn test_from_columns_preserves_arc_sharing() {
    let s = Series::new::<i32>("x", &[1, 2, 3]).unwrap();
    let arc_before = s.column_arc().clone();

    let df = DataFrame::from_columns([("x", s)]).unwrap();
    let retrieved = df.series("x").unwrap();

    // The Arc inside the DataFrame should be the same Arc as the input Series.
    assert!(Arc::ptr_eq(&arc_before, retrieved.column_arc()));
}
