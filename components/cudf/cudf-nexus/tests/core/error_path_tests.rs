use cudf::{ErrorCode, ErrorKind, core::Table, testing::prelude::*};

use crate::common::{column_from_slice, error_context_value};

#[test]
fn test_dataframe_new_mismatched_column_lengths() {
    let c1 = column_from_slice(&[1i32, 2, 3]).unwrap();
    let c2 = column_from_slice(&[1.0f64, 2.0]).unwrap(); // length 2 vs 3
    let result = crate::common::dataframe_from_columns(vec!["a".to_string(), "b".to_string()], vec![c1, c2]);
    assert!(result.is_err());
}

#[test]
fn test_dataframe_series_miss_returns_err() {
    let c1 = column_from_slice(&[1i32, 2, 3]).unwrap();
    let df = crate::common::dataframe_from_columns(vec!["a".to_string()], vec![c1]).unwrap();
    let result = df.series("nonexistent");
    assert!(result.is_err());
}

#[test]
fn test_filter_mask_length_mismatch() {
    let col = column_from_slice(&[1i32, 2, 3]).unwrap();
    let mask = column_from_slice(&[true, false]).unwrap(); // length 2 vs 3
    let table = Table::from_columns(vec![col]).unwrap();
    let mask_view = mask.testing_view().unwrap();
    let result = cudf::ops::filter::apply_boolean_mask(&table.testing_view().unwrap(), &mask_view);
    assert!(result.is_err());
}

#[test]
fn test_parquet_read_nonexistent_file() {
    let rt = crate::common::sample_runtime();
    let options = cudf::io::parquet::ParquetTableReadOptions::builder()
        .filepath("/tmp/nonexistent_cudf_test_file_12345.parquet")
        .with_runtime(&rt)
        .build();
    let result = cudf::io::parquet::read_parquet_table(&options);
    assert!(result.is_err());
}

#[test]
fn test_concatenate_columns_type_mismatch() {
    let a = column_from_slice(&[1i32, 2]).unwrap();
    let b = column_from_slice(&[1.0f64, 2.0]).unwrap();
    let a_view = a.testing_view().unwrap();
    let b_view = b.testing_view().unwrap();
    let result = cudf::ops::concatenate::concatenate_columns(&[&a_view, &b_view]);
    assert!(result.is_err());
}

#[test]
fn test_concatenate_tables_column_count_mismatch() {
    let a = column_from_slice(&[1i32, 2]).unwrap();
    let b = column_from_slice(&[3i32, 4]).unwrap();
    let c = column_from_slice(&[5i32, 6]).unwrap();
    let t1 = Table::from_columns(vec![a, b]).unwrap();
    let t2 = Table::from_columns(vec![c]).unwrap();
    let t1_view = t1.testing_view().unwrap();
    let t2_view = t2.testing_view().unwrap();
    let result = cudf::ops::concatenate::concatenate_tables(&[&t1_view, &t2_view]);
    assert!(result.is_err());
}

#[test]
fn test_dataframe_new_names_columns_count_mismatch() {
    let c1 = column_from_slice(&[1i32, 2, 3]).unwrap();
    let result = crate::common::dataframe_from_columns(vec!["a".to_string(), "b".to_string()], vec![c1]);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.code(), ErrorCode::ColumnLengthMismatch);
    assert_eq!(err.operation(), "dataframe::new");
    assert_eq!(error_context_value(&err, "column_name_count").as_deref(), Some("2"));
    assert_eq!(error_context_value(&err, "table_column_count").as_deref(), Some("1"));
    assert_eq!(error_context_value(&err, "names_count").as_deref(), None);
    assert_eq!(error_context_value(&err, "columns_count").as_deref(), None);
}

#[test]
fn test_dataframe_new_duplicate_names() {
    let c1 = column_from_slice(&[1i32, 2]).unwrap();
    let c2 = column_from_slice(&[3i32, 4]).unwrap();
    let result = crate::common::dataframe_from_columns(vec!["a".to_string(), "a".to_string()], vec![c1, c2]);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
}
