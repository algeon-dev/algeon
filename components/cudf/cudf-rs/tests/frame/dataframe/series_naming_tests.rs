//! Tests for Series naming authority in DataFrame construction.
//!
//! Validates that `DataFrame::from_series` uses Series-carried names,
//! and `DataFrame::from_named_pairs` provides explicit schema override.

use cudf::{DataFrame, ErrorKind, Series};

// ---------------------------------------------------------------------------
// from_series: Series-carried names
// ---------------------------------------------------------------------------

#[test]
fn test_from_series_uses_series_names() {
    let x = Series::new::<i32>("x", &[1, 2, 3]).unwrap();
    let y = Series::new::<f64>("y", &[10.0, 20.0, 30.0]).unwrap();

    let df = DataFrame::from_series([x, y]).unwrap();

    assert_eq!(df.num_columns(), 2);
    assert_eq!(df.num_rows(), 3);
    assert_eq!(df.column_names(), vec!["x", "y"]);
}

#[test]
fn test_from_series_rejects_unnamed() {
    let named = Series::new::<i32>("a", &[1, 2]).unwrap();
    let unnamed = Series::from_column(cudf::core::Column::from_slice(&[3i32, 4]).unwrap()).unwrap();

    let err = DataFrame::from_series([named, unnamed]).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
}

#[test]
fn test_from_series_rejects_duplicate_names() {
    let a1 = Series::new::<i32>("dup", &[1, 2]).unwrap();
    let a2 = Series::new::<i32>("dup", &[3, 4]).unwrap();

    let err = DataFrame::from_series([a1, a2]).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
}

#[test]
fn test_from_series_empty_iterator() {
    let df = DataFrame::from_series(Vec::<Series>::new()).unwrap();
    assert_eq!(df.num_columns(), 0);
    assert_eq!(df.num_rows(), 0);
}

#[test]
fn test_from_series_mismatched_lengths() {
    let a = Series::new::<i32>("a", &[1, 2, 3]).unwrap();
    let b = Series::new::<i32>("b", &[1, 2]).unwrap();

    let err = DataFrame::from_series([a, b]).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
}

// ---------------------------------------------------------------------------
// from_named_pairs: explicit schema override
// ---------------------------------------------------------------------------

#[test]
fn test_from_named_pairs_basic() {
    let x = Series::new::<i32>("source_x", &[1, 2, 3]).unwrap();
    let y = Series::new::<f64>("source_y", &[10.0, 20.0, 30.0]).unwrap();

    // Schema key overrides the Series' own name.
    let df = DataFrame::from_named_pairs([("x", x), ("y", y)]).unwrap();

    assert_eq!(df.num_columns(), 2);
    assert_eq!(df.column_names(), vec!["x", "y"]);
}

#[test]
fn test_from_named_pairs_rejects_duplicates() {
    let a = Series::new::<i32>("a", &[1]).unwrap();
    let b = Series::new::<i32>("b", &[2]).unwrap();

    let err = DataFrame::from_named_pairs([("same", a), ("same", b)]).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
}

// ---------------------------------------------------------------------------
// dataframe! macro
// ---------------------------------------------------------------------------

#[test]
fn test_dataframe_macro_uses_from_named_pairs() {
    let x = Series::new::<i32>("src", &[1, 2]).unwrap();
    let y = Series::new::<f64>("src", &[3.0, 4.0]).unwrap();

    let df = cudf::dataframe! { "a" => x, "b" => y }.unwrap();
    assert_eq!(df.column_names(), vec!["a", "b"]);
}
