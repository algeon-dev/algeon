//! Tests for the DataFrame wrapper.

use cudf::{DataFrame, ErrorCode, ErrorKind, core::Table, testing::prelude::*};

use crate::common::{assert_duplicate_name_error, assert_missing_column_error, column_from_slice, error_context_value};

// ---------------------------------------------------------------------------
// Construction and column-by-name access
// ---------------------------------------------------------------------------

#[test]
fn test_new_from_named_columns() {
    let ids = column_from_slice(&[1i32, 2, 3]).unwrap();
    let scores = column_from_slice(&[10.0f64, 20.0, 30.0]).unwrap();

    let df =
        crate::common::dataframe_from_columns(vec!["id".to_string(), "score".to_string()], vec![ids, scores]).unwrap();

    assert_eq!(df.num_columns(), 2);
    assert_eq!(df.num_rows(), 3);
}

#[test]
fn test_column_by_name() {
    let ids = column_from_slice(&[1i32, 2, 3]).unwrap();
    let scores = column_from_slice(&[10.0f64, 20.0, 30.0]).unwrap();

    let df =
        crate::common::dataframe_from_columns(vec!["id".to_string(), "score".to_string()], vec![ids, scores]).unwrap();

    let col = df.series("id").unwrap();
    assert_eq!(col.to_vec_strict::<i32>().unwrap(), vec![1, 2, 3]);

    let col = df.series("score").unwrap();
    assert_eq!(col.to_vec_strict::<f64>().unwrap(), vec![10.0, 20.0, 30.0]);
}

#[test]
fn test_reject_duplicate_column_names() {
    let c1 = column_from_slice(&[1i32, 2]).unwrap();
    let c2 = column_from_slice(&[3i32, 4]).unwrap();

    let result = crate::common::dataframe_from_columns(vec!["x".to_string(), "x".to_string()], vec![c1, c2]);

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert_duplicate_name_error(&err, "x");
    assert_eq!(err.operation(), "dataframe::new");
    assert_eq!(error_context_value(&err, "column_name_count"), Some("2"));
}

#[test]
fn test_reject_mismatched_names_columns_count() {
    let c1 = column_from_slice(&[1i32, 2]).unwrap();

    let result = crate::common::dataframe_from_columns(vec!["a".to_string(), "b".to_string()], vec![c1]);

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.code(), ErrorCode::ColumnLengthMismatch);
    assert_eq!(err.operation(), "dataframe::new");
    assert_eq!(error_context_value(&err, "column_name_count"), Some("2"));
    assert_eq!(error_context_value(&err, "table_column_count"), Some("1"));
    assert_eq!(error_context_value(&err, "names_count"), None);
    assert_eq!(error_context_value(&err, "columns_count"), None);
}

#[test]
fn test_column_nonexistent_name_returns_error() {
    let c1 = column_from_slice(&[1i32, 2]).unwrap();
    let df = crate::common::dataframe_from_columns(vec!["x".to_string()], vec![c1]).unwrap();

    let result = df.series("nonexistent");
    assert!(result.is_err());
    let err = result.err().unwrap();
    assert_missing_column_error(&err, "nonexistent");
}

#[test]
fn test_column_names_preserves_insertion_order() {
    let c1 = column_from_slice(&[1i32]).unwrap();
    let c2 = column_from_slice(&[2i32]).unwrap();
    let c3 = column_from_slice(&[3i32]).unwrap();

    let df = crate::common::dataframe_from_columns(
        vec!["zebra".to_string(), "apple".to_string(), "mango".to_string()],
        vec![c1, c2, c3],
    )
    .unwrap();

    let names = df.column_names();
    assert_eq!(names, &["zebra", "apple", "mango"]);
}

#[test]
fn test_column_index() {
    let c1 = column_from_slice(&[1i32]).unwrap();
    let c2 = column_from_slice(&[2i32]).unwrap();

    let df = crate::common::dataframe_from_columns(vec!["a".to_string(), "b".to_string()], vec![c1, c2]).unwrap();

    assert_eq!(df.column_index("a"), Some(0));
    assert_eq!(df.column_index("b"), Some(1));
    assert_eq!(df.column_index("c"), None);
}

#[test]
fn test_column_names_and_index_give_equivalent_info() {
    let c1 = column_from_slice(&[1i32]).unwrap();
    let c2 = column_from_slice(&[2.0f64]).unwrap();

    let df = crate::common::dataframe_from_columns(vec!["id".to_string(), "val".to_string()], vec![c1, c2]).unwrap();

    let names = df.column_names();
    assert_eq!(names.len(), 2);
    assert_eq!(names[0], "id");
    assert_eq!(names[1], "val");
    assert_eq!(df.column_index("id"), Some(0));
    assert_eq!(df.column_index("val"), Some(1));
}

#[test]
fn test_column_accessor_validates_data() {
    let c1 = column_from_slice(&[1i32, 2]).unwrap();
    let df = crate::common::dataframe_from_columns(vec!["x".to_string()], vec![c1]).unwrap();

    let col = df.series("x").unwrap();
    assert_eq!(col.to_vec_strict::<i32>().unwrap(), vec![1, 2]);
    assert_eq!(df.num_columns(), 1);
    assert_eq!(df.num_rows(), 2);
}

#[test]
fn test_from_table_round_trip() {
    // Build a Table directly, then wrap it in a DataFrame.
    let c1 = column_from_slice(&[10i32, 20, 30]).unwrap();
    let c2 = column_from_slice(&[1.5f64, 2.5, 3.5]).unwrap();
    let table = Table::from_columns(vec![c1, c2]).unwrap();

    let df = DataFrame::from_table(table, vec!["a".to_string(), "b".to_string()]).unwrap();

    assert_eq!(df.num_columns(), 2);
    assert_eq!(df.num_rows(), 3);

    let col_a = df.series("a").unwrap();
    assert_eq!(col_a.to_vec_strict::<i32>().unwrap(), vec![10, 20, 30]);

    let col_b = df.series("b").unwrap();
    assert_eq!(col_b.to_vec_strict::<f64>().unwrap(), vec![1.5, 2.5, 3.5]);
}

#[test]
fn test_from_table_mismatched_names_count() {
    let c1 = column_from_slice(&[1i32]).unwrap();
    let table = Table::from_columns(vec![c1]).unwrap();

    let result = DataFrame::from_table(table, vec!["a".to_string(), "b".to_string()]);

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.code(), ErrorCode::ColumnLengthMismatch);
    assert_eq!(err.operation(), "dataframe::from_table");
    assert_eq!(error_context_value(&err, "column_name_count"), Some("2"));
    assert_eq!(error_context_value(&err, "table_column_count"), Some("1"));
}

#[test]
fn test_into_table() {
    let c1 = column_from_slice(&[1i32, 2]).unwrap();
    let c2 = column_from_slice(&[3.0f64, 4.0]).unwrap();
    let df = crate::common::dataframe_from_columns(vec!["x".to_string(), "y".to_string()], vec![c1, c2]).unwrap();

    let (table, names) = df.try_into_table().unwrap();
    assert_eq!(names, vec!["x", "y"]);

    let view = table.testing_view().unwrap();
    assert_eq!(view.num_columns().unwrap(), 2);
    assert_eq!(view.num_rows().unwrap(), 2);
}

#[test]
fn test_empty_dataframe() {
    let df = crate::common::dataframe_from_columns(vec![], vec![]).unwrap();
    assert_eq!(df.num_columns(), 0);
    assert!(df.column_names().is_empty());
}

// ---------------------------------------------------------------------------
// dataframe! macro
// ---------------------------------------------------------------------------

#[test]
fn test_dataframe_macro_basic() {
    let x = crate::common::series::<i32>("source_x", &[1, 2, 3]).unwrap();
    let y = crate::common::series::<f64>("source_y", &[10.0, 20.0, 30.0]).unwrap();

    let df = cudf::dataframe! { "x" => x, "y" => y }.unwrap();

    assert_eq!(df.num_columns(), 2);
    assert_eq!(df.column_names(), &["x", "y"]);
    assert_eq!(df.num_rows(), 3);
    assert_eq!(df.series("x").unwrap().to_vec_strict::<i32>().unwrap(), vec![1, 2, 3]);
}

#[test]
fn test_dataframe_macro_single_column() {
    let series = crate::common::series::<i32>("source_only", &[1, 2]).unwrap();

    let df = cudf::dataframe! { "only" => series }.unwrap();

    assert_eq!(df.num_columns(), 1);
    assert_eq!(df.column_names(), &["only"]);
}

#[test]
fn test_dataframe_macro_trailing_comma() {
    let a = crate::common::series::<i32>("source_a", &[1]).unwrap();
    let b = crate::common::series::<i32>("source_b", &[2]).unwrap();

    let df = cudf::dataframe! {
        "a" => a,
        "b" => b,
    }
    .unwrap();

    assert_eq!(df.num_columns(), 2);
    assert_eq!(df.column_names(), &["a", "b"]);
}

#[test]
fn test_dataframe_macro_uses_pair_names_not_series_names() {
    let source = crate::common::series::<i32>("upstream_name", &[11, 22]).unwrap();

    let df = cudf::dataframe! { "renamed" => source }.unwrap();

    assert_eq!(df.column_names(), &["renamed"]);
}
