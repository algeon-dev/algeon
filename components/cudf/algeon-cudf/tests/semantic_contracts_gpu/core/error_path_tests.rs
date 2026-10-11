use cudf::{
    ErrorCode, ErrorKind,
    io::parquet::{ParquetTableReadOptions, read_parquet_table},
};

use crate::common::{column_from_slice, dataframe_from_columns, error_context_value, sample_runtime};

fn test_dataframe_new_mismatched_column_lengths() {
    let c1 = column_from_slice(&[1i32, 2, 3]).unwrap();
    let c2 = column_from_slice(&[1.0f64, 2.0]).unwrap(); // length 2 vs 3
    let result = dataframe_from_columns(vec!["a".to_string(), "b".to_string()], vec![c1, c2]);
    assert!(result.is_err());
}

fn test_dataframe_series_miss_returns_err() {
    let c1 = column_from_slice(&[1i32, 2, 3]).unwrap();
    let df = dataframe_from_columns(vec!["a".to_string()], vec![c1]).unwrap();
    let result = df.series("nonexistent");
    assert!(result.is_err());
}

fn test_parquet_read_nonexistent_file() {
    let rt = sample_runtime();
    let options = ParquetTableReadOptions::builder()
        .filepath("/tmp/nonexistent_cudf_test_file_12345.parquet")
        .with_runtime(&rt)
        .build();
    let result = read_parquet_table(&options);
    assert!(result.is_err());
}

fn test_dataframe_new_names_columns_count_mismatch() {
    let c1 = column_from_slice(&[1i32, 2, 3]).unwrap();
    let result = dataframe_from_columns(vec!["a".to_string(), "b".to_string()], vec![c1]);
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

fn test_dataframe_new_duplicate_names() {
    let c1 = column_from_slice(&[1i32, 2]).unwrap();
    let c2 = column_from_slice(&[3i32, 4]).unwrap();
    let result = dataframe_from_columns(vec!["a".to_string(), "a".to_string()], vec![c1, c2]);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
}

pub(super) const CASES: &[(&str, fn())] = &[
    ("error_path_tests::test_dataframe_new_mismatched_column_lengths", test_dataframe_new_mismatched_column_lengths),
    ("error_path_tests::test_dataframe_series_miss_returns_err", test_dataframe_series_miss_returns_err),
    ("error_path_tests::test_parquet_read_nonexistent_file", test_parquet_read_nonexistent_file),
    (
        "error_path_tests::test_dataframe_new_names_columns_count_mismatch",
        test_dataframe_new_names_columns_count_mismatch,
    ),
    ("error_path_tests::test_dataframe_new_duplicate_names", test_dataframe_new_duplicate_names),
];
