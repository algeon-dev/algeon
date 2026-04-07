//! Tests for DataFrame I/O integration.

use cudf::{
    CsvReadOptions, CsvWriteOptions, DataFrame, ParquetReaderOptions, Series,
    core::{Column, Table},
};
use tempfile::TempDir;

#[test]
fn test_read_parquet_returns_dataframe_with_schema() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test_read.parquet");

    // Write a parquet file with known column names using the low-level API.
    let c1 = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let c2 = Column::from_slice(&[10.0f64, 20.0, 30.0]).unwrap();
    let table = Table::from_columns(vec![c1, c2]).unwrap();
    let view = table.view().unwrap();
    let write_opts =
        cudf::io::parquet::ParquetWriterOptions::builder().filepath(&path).column_names(&["id", "value"]).build();
    cudf::io::parquet::write_parquet(&view, &write_opts).unwrap();

    // Read back via DataFrame.
    let df = DataFrame::read_parquet(&path).unwrap();

    assert_eq!(df.num_columns(), 2);
    assert_eq!(df.num_rows(), 3);
    assert_eq!(df.column_names(), vec!["id", "value"]);

    let col = df.series("id").unwrap();
    assert_eq!(col.to_vec::<i32>().unwrap(), vec![1, 2, 3]);

    let col = df.series("value").unwrap();
    assert_eq!(col.to_vec::<f64>().unwrap(), vec![10.0, 20.0, 30.0]);
}

#[test]
fn test_read_parquet_with_options() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test_opts.parquet");

    let c1 = Column::from_slice(&[1i32, 2, 3, 4, 5]).unwrap();
    let c2 = Column::from_slice(&[10i32, 20, 30, 40, 50]).unwrap();
    let table = Table::from_columns(vec![c1, c2]).unwrap();
    let view = table.view().unwrap();
    let write_opts =
        cudf::io::parquet::ParquetWriterOptions::builder().filepath(&path).column_names(&["a", "b"]).build();
    cudf::io::parquet::write_parquet(&view, &write_opts).unwrap();

    // Read with column selection and row limit.
    let opts = ParquetReaderOptions::builder().filepath(&path).columns(&["b"]).num_rows(3).build();
    let df = DataFrame::read_parquet_with_options(opts).unwrap();

    assert_eq!(df.num_columns(), 1);
    assert_eq!(df.num_rows(), 3);
    assert_eq!(df.column_names(), vec!["b"]);

    let col = df.series("b").unwrap();
    assert_eq!(col.to_vec::<i32>().unwrap(), vec![10, 20, 30]);
}

#[test]
fn test_write_parquet_preserves_schema() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test_write.parquet");

    let c1 = Column::from_slice(&[100i32, 200]).unwrap();
    let c2 = Column::from_slice(&[1.5f64, 2.5]).unwrap();
    let df = DataFrame::new(vec!["x".to_string(), "y".to_string()], vec![c1, c2]).unwrap();

    df.write_parquet(&path).unwrap();

    // Read back and verify schema preserved.
    let df2 = DataFrame::read_parquet(&path).unwrap();
    assert_eq!(df2.column_names(), vec!["x", "y"]);
    assert_eq!(df2.num_rows(), 2);

    let col = df2.series("x").unwrap();
    assert_eq!(col.to_vec::<i32>().unwrap(), vec![100, 200]);

    let col = df2.series("y").unwrap();
    assert_eq!(col.to_vec::<f64>().unwrap(), vec![1.5, 2.5]);
}

#[test]
fn test_roundtrip_create_write_read() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("roundtrip.parquet");

    // Create DataFrame → write → read → verify identical.
    let c1 = Column::from_slice(&[10i64, 20, 30]).unwrap();
    let c2 = Column::from_slice(&[1.0f32, 2.0, 3.0]).unwrap();
    let c3 = Column::from_slice(&[7i32, 8, 9]).unwrap();
    let df =
        DataFrame::new(vec!["alpha".to_string(), "beta".to_string(), "gamma".to_string()], vec![c1, c2, c3]).unwrap();

    df.write_parquet(&path).unwrap();

    let df2 = DataFrame::read_parquet(&path).unwrap();
    assert_eq!(df2.column_names(), vec!["alpha", "beta", "gamma"]);
    assert_eq!(df2.num_columns(), 3);
    assert_eq!(df2.num_rows(), 3);

    assert_eq!(df2.series("alpha").unwrap().to_vec::<i64>().unwrap(), vec![10, 20, 30]);
    assert_eq!(df2.series("beta").unwrap().to_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
    assert_eq!(df2.series("gamma").unwrap().to_vec::<i32>().unwrap(), vec![7, 8, 9]);
}

#[test]
fn test_read_csv_returns_dataframe_with_schema() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test_read.csv");

    std::fs::write(&path, "id,value\n1,10\n2,20\n3,30\n").unwrap();

    let df = DataFrame::read_csv(&path).unwrap();

    assert_eq!(df.num_columns(), 2);
    assert_eq!(df.num_rows(), 3);
    assert_eq!(df.column_names(), vec!["id", "value"]);
    assert_eq!(df.series("id").unwrap().to_vec::<i64>().unwrap(), vec![1, 2, 3]);
    assert_eq!(df.series("value").unwrap().to_vec::<i64>().unwrap(), vec![10, 20, 30]);
}

#[test]
fn test_read_csv_with_options_supports_delimiter_and_explicit_names() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test_opts.csv");

    std::fs::write(&path, "1\t10\n2\t20\n3\t30\n").unwrap();

    let opts = CsvReadOptions::builder()
        .filepath(&path)
        .delimiter(b'\t')
        .has_header(false)
        .column_names(&["left", "right"])
        .build();
    let df = DataFrame::read_csv_with_options(opts).unwrap();

    assert_eq!(df.num_columns(), 2);
    assert_eq!(df.num_rows(), 3);
    assert_eq!(df.column_names(), vec!["left", "right"]);
    assert_eq!(df.series("left").unwrap().to_vec::<i64>().unwrap(), vec![1, 2, 3]);
    assert_eq!(df.series("right").unwrap().to_vec::<i64>().unwrap(), vec![10, 20, 30]);
}

#[test]
fn test_read_csv_without_header_uses_generated_names() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test_no_header.csv");

    std::fs::write(&path, "4,40\n5,50\n").unwrap();

    let opts = CsvReadOptions::builder().filepath(&path).has_header(false).build();
    let df = DataFrame::read_csv_with_options(opts).unwrap();

    assert_eq!(df.column_names(), vec!["column_0", "column_1"]);
    assert_eq!(df.series("column_0").unwrap().to_vec::<i64>().unwrap(), vec![4, 5]);
    assert_eq!(df.series("column_1").unwrap().to_vec::<i64>().unwrap(), vec![40, 50]);
}

#[test]
fn test_read_csv_without_header_generates_fallback_names_only_at_dataframe_layer() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test_no_header_contract.csv");

    std::fs::write(&path, "11,110\n12,120\n").unwrap();

    let table_result =
        cudf::io::csv::read_csv(&cudf::io::csv::CsvReaderOptions::builder().filepath(&path).has_header(false).build())
            .unwrap();
    assert_eq!(table_result.column_names, None);

    let df =
        DataFrame::read_csv_with_options(CsvReadOptions::builder().filepath(&path).has_header(false).build()).unwrap();

    assert_eq!(df.column_names(), vec!["column_0", "column_1"]);
    assert_eq!(df.series("column_0").unwrap().to_vec::<i64>().unwrap(), vec![11, 12]);
    assert_eq!(df.series("column_1").unwrap().to_vec::<i64>().unwrap(), vec![110, 120]);
}

#[test]
fn test_read_csv_explicit_names_override_header_names() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test_override.csv");

    std::fs::write(&path, "old_a,old_b\n7,70\n8,80\n").unwrap();

    let opts = CsvReadOptions::builder().filepath(&path).column_names(&["new_a", "new_b"]).build();
    let df = DataFrame::read_csv_with_options(opts).unwrap();

    assert_eq!(df.column_names(), vec!["new_a", "new_b"]);
    assert_eq!(df.series("new_a").unwrap().to_vec::<i64>().unwrap(), vec![7, 8]);
    assert_eq!(df.series("new_b").unwrap().to_vec::<i64>().unwrap(), vec![70, 80]);
}

#[test]
fn test_write_csv_preserves_dataframe_schema() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test_write.csv");

    let ids = Series::new::<i32>("source_ids", &[100, 200]).unwrap();
    let values = Series::new::<i32>("source_values", &[1, 2]).unwrap();
    let df = DataFrame::from_series([("x", ids), ("y", values)]).unwrap();

    df.write_csv(&path).unwrap();

    let written = std::fs::read_to_string(&path).unwrap();
    assert!(written.starts_with("x,y\n"));

    let df2 = DataFrame::read_csv(&path).unwrap();
    assert_eq!(df2.column_names(), vec!["x", "y"]);
    assert_eq!(df2.series("x").unwrap().to_vec::<i64>().unwrap(), vec![100, 200]);
    assert_eq!(df2.series("y").unwrap().to_vec::<i64>().unwrap(), vec![1, 2]);
}

#[test]
fn test_write_csv_with_options_supports_delimiter_and_header_toggle() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test_write_opts.csv");

    let left = Series::new::<i32>("left_src", &[9, 8]).unwrap();
    let right = Series::new::<i32>("right_src", &[90, 80]).unwrap();
    let df = DataFrame::from_series([("left", left), ("right", right)]).unwrap();

    let opts = CsvWriteOptions::builder().filepath(&path).delimiter(b'\t').include_header(false).build();
    df.write_csv_with_options(opts).unwrap();

    let written = std::fs::read_to_string(&path).unwrap();
    assert!(written.starts_with("9\t90\n"));

    let read_opts = CsvReadOptions::builder()
        .filepath(&path)
        .delimiter(b'\t')
        .has_header(false)
        .column_names(&["left", "right"])
        .build();
    let df2 = DataFrame::read_csv_with_options(read_opts).unwrap();

    assert_eq!(df2.column_names(), vec!["left", "right"]);
    assert_eq!(df2.series("left").unwrap().to_vec::<i64>().unwrap(), vec![9, 8]);
    assert_eq!(df2.series("right").unwrap().to_vec::<i64>().unwrap(), vec![90, 80]);
}
