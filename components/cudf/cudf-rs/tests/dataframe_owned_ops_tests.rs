//! Tests for consuming `_owned()` DataFrame method variants.
//!
//! Only schema-only operations have `_owned` variants (rename, select,
//! with_column). GPU-allocating operations (head, tail, filter, sort,
//! drop_nulls) do not have `_owned` variants because FFI always allocates
//! new device memory, making the `Arc::clone` cost negligible.

use cudf::{SortOptions, column::Column, dataframe::DataFrame, error::ErrorKind};

/// Helper: build a 5-row, 2-column DataFrame.
fn sample_df() -> DataFrame {
    let ids = Column::from_slice(&[3i32, 1, 4, 1, 5]).unwrap();
    let scores = Column::from_slice(&[30.0f64, 10.0, 40.0, 15.0, 50.0]).unwrap();
    DataFrame::new(vec!["id".to_string(), "score".to_string()], vec![ids, scores]).unwrap()
}

// ---------------------------------------------------------------------------
// sort_values
// ---------------------------------------------------------------------------

#[test]
fn test_sort_values_ascending() {
    let df = sample_df();
    let sorted = df.sort_values(&["id"], SortOptions::new()).unwrap();

    assert_eq!(sorted.series_view("id").unwrap().to_vec::<i32>().unwrap(), vec![1, 1, 3, 4, 5]);
    assert_eq!(sorted.column_names(), vec!["id", "score"]);
}

#[test]
fn test_sort_values_descending() {
    let df = sample_df();
    let sorted = df.sort_values(&["score"], SortOptions::new().with_descending(vec![true])).unwrap();

    assert_eq!(sorted.series_view("score").unwrap().to_vec::<f64>().unwrap(), vec![50.0, 40.0, 30.0, 15.0, 10.0]);
}

#[test]
fn test_sort_values_consistent_results() {
    let df1 = sample_df();
    let df2 = sample_df();

    let result1 = df1.sort_values(&["id"], SortOptions::new()).unwrap();
    let result2 = df2.sort_values(&["id"], SortOptions::new()).unwrap();

    assert_eq!(
        result1.series_view("id").unwrap().to_vec::<i32>().unwrap(),
        result2.series_view("id").unwrap().to_vec::<i32>().unwrap(),
    );
    assert_eq!(
        result1.series_view("score").unwrap().to_vec::<f64>().unwrap(),
        result2.series_view("score").unwrap().to_vec::<f64>().unwrap(),
    );
}

#[test]
fn test_sort_values_mismatched_descending_count() {
    let df = sample_df();
    let result = df.sort_values(&["id", "score"], SortOptions::new().with_descending(vec![false]));
    assert!(result.is_err());
    assert_eq!(result.unwrap_err().kind(), ErrorKind::InvalidOperation);
}

// ---------------------------------------------------------------------------
// select_owned
// ---------------------------------------------------------------------------

#[test]
fn test_select_owned_basic() {
    let df = sample_df();
    let selected = df.select_owned(&["score"]).unwrap();

    assert_eq!(selected.num_columns(), 1);
    assert_eq!(selected.num_rows(), 5);
    assert_eq!(selected.column_names(), vec!["score"]);
    assert_eq!(selected.series_view("score").unwrap().to_vec::<f64>().unwrap(), vec![30.0, 10.0, 40.0, 15.0, 50.0]);
}

#[test]
fn test_select_owned_reorders_columns() {
    let df = sample_df();
    let selected = df.select_owned(&["score", "id"]).unwrap();

    assert_eq!(selected.column_names(), vec!["score", "id"]);
    assert_eq!(selected.column_index("score"), Some(0));
    assert_eq!(selected.column_index("id"), Some(1));
}

#[test]
fn test_select_owned_nonexistent_column() {
    let df = sample_df();
    let result = df.select_owned(&["missing"]);
    assert!(result.is_err());
    assert_eq!(result.unwrap_err().kind(), ErrorKind::InvalidOperation);
}

#[test]
fn test_select_owned_matches_borrow_variant() {
    let df1 = sample_df();
    let df2 = sample_df();

    let borrow_result = df1.select(&["score"]).unwrap();
    let owned_result = df2.select_owned(&["score"]).unwrap();

    assert_eq!(
        borrow_result.series_view("score").unwrap().to_vec::<f64>().unwrap(),
        owned_result.series_view("score").unwrap().to_vec::<f64>().unwrap(),
    );
}

#[test]
fn test_select_owned_duplicate_column_names_returns_error() {
    let df = sample_df();
    let result = df.select_owned(&["id", "id"]);
    assert!(result.is_err());
    assert_eq!(result.unwrap_err().kind(), ErrorKind::InvalidOperation);
}

#[test]
fn test_select_owned_accepts_string_vec() {
    let df = sample_df();
    let names = vec!["score".to_string(), "id".to_string()];
    let selected = df.select_owned(names).unwrap();
    assert_eq!(selected.column_names(), vec!["score", "id"]);
}

// ---------------------------------------------------------------------------
// with_column_owned
// ---------------------------------------------------------------------------

#[test]
fn test_with_column_owned_adds_new_column() {
    let df = DataFrame::new(vec!["a".to_string()], vec![Column::from_slice(&[1i32, 2, 3]).unwrap()]).unwrap();

    let new_col = Column::from_slice(&[10.0f64, 20.0, 30.0]).unwrap();
    let result = df.with_column_owned("b", new_col).unwrap();

    assert_eq!(result.num_columns(), 2);
    assert_eq!(result.column_names(), vec!["a", "b"]);
    assert_eq!(result.series_view("a").unwrap().to_vec::<i32>().unwrap(), vec![1, 2, 3]);
    assert_eq!(result.series_view("b").unwrap().to_vec::<f64>().unwrap(), vec![10.0, 20.0, 30.0]);
}

#[test]
fn test_with_column_owned_replaces_existing() {
    let df = DataFrame::new(
        vec!["a".to_string(), "b".to_string()],
        vec![Column::from_slice(&[1i32, 2]).unwrap(), Column::from_slice(&[10i32, 20]).unwrap()],
    )
    .unwrap();

    let replacement = Column::from_slice(&[99i32, 100]).unwrap();
    let result = df.with_column_owned("b", replacement).unwrap();

    assert_eq!(result.num_columns(), 2);
    assert_eq!(result.column_names(), vec!["a", "b"]);
    assert_eq!(result.series_view("b").unwrap().to_vec::<i32>().unwrap(), vec![99, 100]);
}

#[test]
fn test_with_column_owned_mismatched_row_count() {
    let df = DataFrame::new(vec!["a".to_string()], vec![Column::from_slice(&[1i32, 2, 3]).unwrap()]).unwrap();

    let new_col = Column::from_slice(&[10i32, 20]).unwrap();
    let err = df.with_column_owned("b", new_col).err().expect("expected row count mismatch");
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
}

// ---------------------------------------------------------------------------
// Method chaining (schema-only _owned methods)
// ---------------------------------------------------------------------------

#[test]
fn test_owned_chain_filter_sort_select() {
    let df = sample_df(); // ids: 3,1,4,1,5  scores: 30,10,40,15,50

    let mask_col = Column::from_slice(&[true, true, true, true, true]).unwrap();
    let mask = mask_col.view().unwrap();

    let result = df
        .filter_by_mask(&mask)
        .unwrap()
        .sort_values(&["score"], SortOptions::new().with_descending(vec![true]))
        .unwrap()
        .select_owned(&["score"])
        .unwrap();

    assert_eq!(result.num_columns(), 1);
    assert_eq!(result.column_names(), vec!["score"]);
    assert_eq!(result.series_view("score").unwrap().to_vec::<f64>().unwrap(), vec![50.0, 40.0, 30.0, 15.0, 10.0]);
}

#[test]
fn test_owned_chain_with_column_then_select() {
    let df = DataFrame::new(
        vec!["a".to_string(), "b".to_string()],
        vec![Column::from_slice(&[1i32, 2, 3]).unwrap(), Column::from_slice(&[10i32, 20, 30]).unwrap()],
    )
    .unwrap();

    let new_col = Column::from_slice(&[100i32, 200, 300]).unwrap();
    let result = df.with_column_owned("c", new_col).unwrap().select_owned(&["a", "c"]).unwrap();

    assert_eq!(result.num_columns(), 2);
    assert_eq!(result.column_names(), vec!["a", "c"]);
    assert_eq!(result.series_view("a").unwrap().to_vec::<i32>().unwrap(), vec![1, 2, 3]);
    assert_eq!(result.series_view("c").unwrap().to_vec::<i32>().unwrap(), vec![100, 200, 300]);
}
