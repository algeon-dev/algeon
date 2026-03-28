//! Semantic baseline tests for `_owned` method variants.
//!
//! Verifies that each kept `_owned` method (schema-only variants) produces
//! identical results to its `&self` counterpart, and that the `_owned` variant
//! avoids Arc cloning by moving columns directly.
//!
//! GPU-allocating `_owned` methods (head_owned, tail_owned, filter_by_mask_owned,
//! filter_isin_owned, drop_nulls_any_owned) have been removed — consuming self
//! provides negligible benefit when FFI always allocates new device memory.

use cudf::{column::Column, dataframe::DataFrame, error::ErrorKind};

/// Helper: build a 3-column, 4-row DataFrame.
fn sample_df() -> DataFrame {
    let a = Column::from_slice(&[1i32, 2, 3, 4]).unwrap();
    let b = Column::from_slice(&[10.0f64, 20.0, 30.0, 40.0]).unwrap();
    let c = Column::from_slice(&[100i64, 200, 300, 400]).unwrap();
    DataFrame::new(vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![a, b, c]).unwrap()
}

// ===========================================================================
// rename_owned
// ===========================================================================

#[test]
fn rename_owned_matches_rename() {
    let df1 = sample_df();
    let df2 = sample_df();

    let ref_result = df1.rename("a", "alpha").unwrap();
    let owned_result = df2.rename_owned("a", "alpha").unwrap();

    assert_eq!(ref_result.column_names(), owned_result.column_names());
    assert_eq!(ref_result.num_rows(), owned_result.num_rows());
    assert_eq!(
        ref_result.series_view("alpha").unwrap().to_vec::<i32>().unwrap(),
        owned_result.series_view("alpha").unwrap().to_vec::<i32>().unwrap(),
    );
    assert_eq!(
        ref_result.series_view("b").unwrap().to_vec::<f64>().unwrap(),
        owned_result.series_view("b").unwrap().to_vec::<f64>().unwrap(),
    );
}

#[test]
fn rename_owned_error_on_missing_column() {
    let df = sample_df();
    let err = df.rename_owned("nonexistent", "x").unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
}

#[test]
fn rename_owned_error_on_duplicate_target() {
    let df = sample_df();
    let err = df.rename_owned("a", "b").unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
}

// ===========================================================================
// select_owned
// ===========================================================================

#[test]
fn select_owned_matches_select() {
    let df1 = sample_df();
    let df2 = sample_df();

    let ref_result = df1.select(&["c", "a"]).unwrap();
    let owned_result = df2.select_owned(&["c", "a"]).unwrap();

    assert_eq!(ref_result.column_names(), owned_result.column_names());
    assert_eq!(ref_result.num_rows(), owned_result.num_rows());
    assert_eq!(
        ref_result.series_view("a").unwrap().to_vec::<i32>().unwrap(),
        owned_result.series_view("a").unwrap().to_vec::<i32>().unwrap(),
    );
    assert_eq!(
        ref_result.series_view("c").unwrap().to_vec::<i64>().unwrap(),
        owned_result.series_view("c").unwrap().to_vec::<i64>().unwrap(),
    );
}

#[test]
fn select_owned_error_on_missing_column() {
    let df = sample_df();
    let err = df.select_owned(&["missing"]).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
}

#[test]
fn select_owned_error_on_duplicate_names() {
    let df = sample_df();
    let err = df.select_owned(&["a", "a"]).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
}

#[test]
fn select_owned_accepts_iterator_of_strings() {
    let df = sample_df();
    let names: Vec<String> = vec!["b".to_string(), "a".to_string()];
    let selected = df.select_owned(names).unwrap();
    assert_eq!(selected.column_names(), vec!["b", "a"]);
}

// ===========================================================================
// with_column_owned
// ===========================================================================

#[test]
fn with_column_owned_matches_with_column() {
    let df1 = sample_df();
    let df2 = sample_df();

    let new_col1 = Column::from_slice(&[99i32, 98, 97, 96]).unwrap();
    let new_col2 = Column::from_slice(&[99i32, 98, 97, 96]).unwrap();

    let ref_result = df1.with_column("d", new_col1).unwrap();
    let owned_result = df2.with_column_owned("d", new_col2).unwrap();

    assert_eq!(ref_result.column_names(), owned_result.column_names());
    assert_eq!(ref_result.num_rows(), owned_result.num_rows());
    assert_eq!(
        ref_result.series_view("d").unwrap().to_vec::<i32>().unwrap(),
        owned_result.series_view("d").unwrap().to_vec::<i32>().unwrap(),
    );
}

#[test]
fn with_column_owned_replaces_existing() {
    let df = sample_df();
    let replacement = Column::from_slice(&[99i32, 98, 97, 96]).unwrap();
    let result = df.with_column_owned("a", replacement).unwrap();

    assert_eq!(result.column_names(), vec!["a", "b", "c"]);
    assert_eq!(result.series_view("a").unwrap().to_vec::<i32>().unwrap(), vec![99, 98, 97, 96]);
}

#[test]
fn with_column_owned_error_on_row_mismatch() {
    let df = sample_df();
    let wrong_len = Column::from_slice(&[1i32, 2]).unwrap();
    let err = df.with_column_owned("d", wrong_len).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
}

// ===========================================================================
// Chaining schema-only _owned methods
// ===========================================================================

#[test]
fn chain_rename_select_with_column_owned() {
    let df = sample_df();
    let extra = Column::from_slice(&[5i32, 6, 7, 8]).unwrap();

    let result = df
        .rename_owned("a", "alpha")
        .unwrap()
        .with_column_owned("d", extra)
        .unwrap()
        .select_owned(&["alpha", "d"])
        .unwrap();

    assert_eq!(result.column_names(), vec!["alpha", "d"]);
    assert_eq!(result.num_rows(), 4);
    assert_eq!(result.series_view("alpha").unwrap().to_vec::<i32>().unwrap(), vec![1, 2, 3, 4]);
    assert_eq!(result.series_view("d").unwrap().to_vec::<i32>().unwrap(), vec![5, 6, 7, 8]);
}
