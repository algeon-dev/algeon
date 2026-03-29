//! Tests for DataFrame concat, copy, rename, select.

use cudf::{column::Column, dataframe::DataFrame, error::ErrorKind};

#[test]
fn test_concat_matching_schemas() {
    let df1 = DataFrame::new(
        vec!["a".to_string(), "b".to_string()],
        vec![Column::from_slice(&[1i32, 2]).unwrap(), Column::from_slice(&[10.0f64, 20.0]).unwrap()],
    )
    .unwrap();

    let df2 = DataFrame::new(
        vec!["a".to_string(), "b".to_string()],
        vec![Column::from_slice(&[3i32, 4]).unwrap(), Column::from_slice(&[30.0f64, 40.0]).unwrap()],
    )
    .unwrap();

    let result = df1.concat(&df2).unwrap();

    assert_eq!(result.num_columns(), 2);
    assert_eq!(result.num_rows(), 4);
    assert_eq!(result.column_names(), vec!["a", "b"]);

    let a_vals: Vec<i32> = result.series_view("a").unwrap().to_vec::<i32>().unwrap();
    assert_eq!(a_vals, vec![1, 2, 3, 4]);

    let b_vals: Vec<f64> = result.series_view("b").unwrap().to_vec::<f64>().unwrap();
    assert_eq!(b_vals, vec![10.0, 20.0, 30.0, 40.0]);
}

#[test]
fn test_concat_mismatched_column_names_returns_error() {
    let df1 = DataFrame::new(
        vec!["a".to_string(), "b".to_string()],
        vec![Column::from_slice(&[1i32, 2]).unwrap(), Column::from_slice(&[10i32, 20]).unwrap()],
    )
    .unwrap();

    let df2 = DataFrame::new(
        vec!["a".to_string(), "c".to_string()],
        vec![Column::from_slice(&[3i32, 4]).unwrap(), Column::from_slice(&[30i32, 40]).unwrap()],
    )
    .unwrap();

    let err = df1.concat(&df2).err().expect("expected schema mismatch error");
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
}

#[test]
fn test_concat_mismatched_column_count_returns_error() {
    let df1 = DataFrame::new(vec!["a".to_string()], vec![Column::from_slice(&[1i32, 2]).unwrap()]).unwrap();

    let df2 = DataFrame::new(
        vec!["a".to_string(), "b".to_string()],
        vec![Column::from_slice(&[3i32, 4]).unwrap(), Column::from_slice(&[30i32, 40]).unwrap()],
    )
    .unwrap();

    let err = df1.concat(&df2).err().expect("expected schema mismatch error");
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
}

#[test]
fn test_concat_mismatched_column_types_returns_error() {
    let df1 = DataFrame::new(
        vec!["a".to_string(), "b".to_string()],
        vec![Column::from_slice(&[1i32, 2]).unwrap(), Column::from_slice(&[10i32, 20]).unwrap()],
    )
    .unwrap();

    // Same schema names/order, but "b" has different type (i64 vs i32).
    let df2 = DataFrame::new(
        vec!["a".to_string(), "b".to_string()],
        vec![Column::from_slice(&[3i32, 4]).unwrap(), Column::from_slice(&[30i64, 40]).unwrap()],
    )
    .unwrap();

    let err = df1.concat(&df2).err().expect("expected type mismatch error");
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
}

#[test]
fn test_copy_preserves_schema_and_data() {
    let df = DataFrame::new(
        vec!["x".to_string(), "y".to_string()],
        vec![Column::from_slice(&[1i32, 2, 3]).unwrap(), Column::from_slice(&[4.0f64, 5.0, 6.0]).unwrap()],
    )
    .unwrap();

    let copy = df.clone();

    assert_eq!(copy.num_columns(), 2);
    assert_eq!(copy.num_rows(), 3);
    assert_eq!(copy.column_names(), vec!["x", "y"]);

    let x_vals: Vec<i32> = copy.series_view("x").unwrap().to_vec::<i32>().unwrap();
    assert_eq!(x_vals, vec![1, 2, 3]);

    let y_vals: Vec<f64> = copy.series_view("y").unwrap().to_vec::<f64>().unwrap();
    assert_eq!(y_vals, vec![4.0, 5.0, 6.0]);
}

#[test]
fn test_rename_column() {
    let df = DataFrame::new(
        vec!["old_name".to_string(), "other".to_string()],
        vec![Column::from_slice(&[1i32, 2]).unwrap(), Column::from_slice(&[10i32, 20]).unwrap()],
    )
    .unwrap();

    let renamed = df.rename_column("old_name", "new_name").unwrap();

    // New name works.
    assert_eq!(renamed.column_names(), vec!["new_name", "other"]);
    let vals: Vec<i32> = renamed.series_view("new_name").unwrap().to_vec::<i32>().unwrap();
    assert_eq!(vals, vec![1, 2]);

    // Old name no longer exists.
    assert!(renamed.series_view("old_name").is_err());

    // Other column is preserved.
    let other_vals: Vec<i32> = renamed.series_view("other").unwrap().to_vec::<i32>().unwrap();
    assert_eq!(other_vals, vec![10, 20]);
}

#[test]
fn test_rename_nonexistent_column_returns_error() {
    let df = DataFrame::new(vec!["a".to_string()], vec![Column::from_slice(&[1i32]).unwrap()]).unwrap();

    let err = df.rename_column("nonexistent", "new").err().expect("expected error");
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
}

#[test]
fn test_rename_to_existing_name_returns_error() {
    let df = DataFrame::new(
        vec!["a".to_string(), "b".to_string()],
        vec![Column::from_slice(&[1i32]).unwrap(), Column::from_slice(&[2i32]).unwrap()],
    )
    .unwrap();

    let err = df.rename_column("a", "b").err().expect("expected error");
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
}

#[test]
fn test_select_subset_of_columns() {
    let df = DataFrame::new(
        vec!["a".to_string(), "b".to_string(), "c".to_string()],
        vec![
            Column::from_slice(&[1i32, 2]).unwrap(),
            Column::from_slice(&[10.0f64, 20.0]).unwrap(),
            Column::from_slice(&[100i64, 200]).unwrap(),
        ],
    )
    .unwrap();

    let selected = df.select(&["c", "a"]).unwrap();

    assert_eq!(selected.num_columns(), 2);
    assert_eq!(selected.num_rows(), 2);
    assert_eq!(selected.column_names(), vec!["c", "a"]);

    let c_vals: Vec<i64> = selected.series_view("c").unwrap().to_vec::<i64>().unwrap();
    assert_eq!(c_vals, vec![100, 200]);

    let a_vals: Vec<i32> = selected.series_view("a").unwrap().to_vec::<i32>().unwrap();
    assert_eq!(a_vals, vec![1, 2]);
}

#[test]
fn test_select_nonexistent_column_returns_error() {
    let df = DataFrame::new(vec!["a".to_string()], vec![Column::from_slice(&[1i32, 2]).unwrap()]).unwrap();

    let err = df.select(&["nonexistent"]).err().expect("expected error");
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
}

#[test]
fn test_select_duplicate_column_names_returns_error() {
    let df = DataFrame::new(
        vec!["a".to_string(), "b".to_string()],
        vec![Column::from_slice(&[1i32, 2]).unwrap(), Column::from_slice(&[10i32, 20]).unwrap()],
    )
    .unwrap();

    let err = df.select(&["a", "a"]).err().expect("expected duplicate selection error");
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
}

#[test]
fn test_select_preserves_data() {
    let df = DataFrame::new(
        vec!["x".to_string(), "y".to_string()],
        vec![Column::from_slice(&[1i32, 2, 3]).unwrap(), Column::from_slice(&[4.0f64, 5.0, 6.0]).unwrap()],
    )
    .unwrap();

    // Select single column.
    let selected = df.select(&["y"]).unwrap();

    assert_eq!(selected.num_columns(), 1);
    assert_eq!(selected.column_names(), vec!["y"]);

    let y_vals: Vec<f64> = selected.series_view("y").unwrap().to_vec::<f64>().unwrap();
    assert_eq!(y_vals, vec![4.0, 5.0, 6.0]);
}

#[test]
fn test_select_accepts_string_vec() {
    let df = DataFrame::new(
        vec!["a".to_string(), "b".to_string()],
        vec![Column::from_slice(&[1i32, 2]).unwrap(), Column::from_slice(&[10i32, 20]).unwrap()],
    )
    .unwrap();

    let names = vec!["b".to_string(), "a".to_string()];
    let selected = df.select(names).unwrap();

    assert_eq!(selected.column_names(), vec!["b", "a"]);
}

#[test]
fn test_rename_preserves_column_order() {
    let df = DataFrame::new(
        vec!["a".to_string(), "b".to_string(), "c".to_string()],
        vec![
            Column::from_slice(&[1i32]).unwrap(),
            Column::from_slice(&[2i32]).unwrap(),
            Column::from_slice(&[3i32]).unwrap(),
        ],
    )
    .unwrap();

    let renamed = df.rename_column("b", "beta").unwrap();
    assert_eq!(renamed.column_names(), vec!["a", "beta", "c"]);
}

#[test]
fn test_rename_same_name_is_schema_only_and_preserves_data() {
    let df = DataFrame::new(
        vec!["a".to_string(), "b".to_string()],
        vec![Column::from_slice(&[1i32, 2, 3]).unwrap(), Column::from_slice(&[10i32, 20, 30]).unwrap()],
    )
    .unwrap();

    let renamed = df.rename_column("a", "a").unwrap();

    // No schema change.
    assert_eq!(renamed.column_names(), vec!["a", "b"]);

    // Data unchanged.
    assert_eq!(renamed.series_view("a").unwrap().to_vec::<i32>().unwrap(), vec![1, 2, 3]);
    assert_eq!(renamed.series_view("b").unwrap().to_vec::<i32>().unwrap(), vec![10, 20, 30]);
}

#[test]
fn test_select_all_columns_returns_full_copy() {
    let df = DataFrame::new(
        vec!["a".to_string(), "b".to_string()],
        vec![Column::from_slice(&[1i32, 2]).unwrap(), Column::from_slice(&[3i32, 4]).unwrap()],
    )
    .unwrap();

    let selected = df.select(&["a", "b"]).unwrap();
    assert_eq!(selected.num_columns(), 2);
    assert_eq!(selected.num_rows(), 2);
    assert_eq!(selected.column_names(), vec!["a", "b"]);

    let a_vals: Vec<i32> = selected.series_view("a").unwrap().to_vec::<i32>().unwrap();
    assert_eq!(a_vals, vec![1, 2]);
}
