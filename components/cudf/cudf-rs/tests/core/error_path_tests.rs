use cudf::{
    ErrorKind,
    core::{Column, Table},
};

#[test]
fn test_dataframe_new_mismatched_column_lengths() {
    let c1 = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let c2 = Column::from_slice(&[1.0f64, 2.0]).unwrap(); // length 2 vs 3
    let result = crate::common::dataframe_from_columns(vec!["a".to_string(), "b".to_string()], vec![c1, c2]);
    assert!(result.is_err());
}

#[test]
fn test_dataframe_series_miss_returns_err() {
    let c1 = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let df = crate::common::dataframe_from_columns(vec!["a".to_string()], vec![c1]).unwrap();
    let result = df.series("nonexistent");
    assert!(result.is_err());
}

#[test]
fn test_filter_mask_length_mismatch() {
    let col = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let mask = Column::from_slice(&[true, false]).unwrap(); // length 2 vs 3
    let table = Table::from_columns(vec![col]).unwrap();
    let mask_view = mask.view().unwrap();
    let result = cudf::ops::filter::apply_boolean_mask(&table.view().unwrap(), &mask_view);
    assert!(result.is_err());
}

#[test]
fn test_csv_read_nonexistent_file() {
    let options =
        cudf::io::csv::CsvTableReadOptions::builder().filepath("/tmp/nonexistent_cudf_test_file_12345.csv").build();
    let result = cudf::io::csv::read_csv(&options);
    assert!(result.is_err());
}

#[test]
fn test_parquet_read_nonexistent_file() {
    let options = cudf::io::parquet::ParquetReadOptions::builder()
        .filepath("/tmp/nonexistent_cudf_test_file_12345.parquet")
        .build();
    let result = cudf::io::parquet::read_parquet(&options);
    assert!(result.is_err());
}

#[test]
fn test_concatenate_columns_type_mismatch() {
    let a = Column::from_slice(&[1i32, 2]).unwrap();
    let b = Column::from_slice(&[1.0f64, 2.0]).unwrap();
    let a_view = a.view().unwrap();
    let b_view = b.view().unwrap();
    let result = cudf::ops::concatenate::concatenate_columns(&[&a_view, &b_view]);
    assert!(result.is_err());
}

#[test]
fn test_concatenate_tables_column_count_mismatch() {
    let a = Column::from_slice(&[1i32, 2]).unwrap();
    let b = Column::from_slice(&[3i32, 4]).unwrap();
    let c = Column::from_slice(&[5i32, 6]).unwrap();
    let t1 = Table::from_columns(vec![a, b]).unwrap();
    let t2 = Table::from_columns(vec![c]).unwrap();
    let t1_view = t1.view().unwrap();
    let t2_view = t2.view().unwrap();
    let result = cudf::ops::concatenate::concatenate_tables(&[&t1_view, &t2_view]);
    assert!(result.is_err());
}

#[test]
fn test_dataframe_new_names_columns_count_mismatch() {
    let c1 = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let result = crate::common::dataframe_from_columns(vec!["a".to_string(), "b".to_string()], vec![c1]);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
}

#[test]
fn test_dataframe_new_duplicate_names() {
    let c1 = Column::from_slice(&[1i32, 2]).unwrap();
    let c2 = Column::from_slice(&[3i32, 4]).unwrap();
    let result = crate::common::dataframe_from_columns(vec!["a".to_string(), "a".to_string()], vec![c1, c2]);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
}
