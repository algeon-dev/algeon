//! Tests for DataFrame I/O integration.

use cudf::{column::Column, dataframe::DataFrame, io::parquet::ParquetReaderOptions};
use tempfile::TempDir;

#[test]
fn test_read_parquet_returns_dataframe_with_schema() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test_read.parquet");

    // Write a parquet file with known column names using the low-level API.
    let c1 = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let c2 = Column::from_slice(&[10.0f64, 20.0, 30.0]).unwrap();
    let table = cudf::table::Table::from_columns(vec![c1, c2]).unwrap();
    let view = table.view().unwrap();
    let write_opts = cudf::io::parquet::ParquetWriterOptions::new(&path).column_names(&["id", "value"]).unwrap();
    cudf::io::parquet::write_parquet(&view, &write_opts).unwrap();

    // Read back via DataFrame.
    let df = DataFrame::read_parquet(&path).unwrap();

    assert_eq!(df.num_columns(), 2);
    assert_eq!(df.num_rows(), 3);
    assert_eq!(df.column_names(), vec!["id", "value"]);

    let col = df.series_view("id").unwrap();
    assert_eq!(col.to_vec::<i32>().unwrap(), vec![1, 2, 3]);

    let col = df.series_view("value").unwrap();
    assert_eq!(col.to_vec::<f64>().unwrap(), vec![10.0, 20.0, 30.0]);
}

#[test]
fn test_read_parquet_with_options() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test_opts.parquet");

    let c1 = Column::from_slice(&[1i32, 2, 3, 4, 5]).unwrap();
    let c2 = Column::from_slice(&[10i32, 20, 30, 40, 50]).unwrap();
    let table = cudf::table::Table::from_columns(vec![c1, c2]).unwrap();
    let view = table.view().unwrap();
    let write_opts = cudf::io::parquet::ParquetWriterOptions::new(&path).column_names(&["a", "b"]).unwrap();
    cudf::io::parquet::write_parquet(&view, &write_opts).unwrap();

    // Read with column selection and row limit.
    let opts = ParquetReaderOptions::new(&path).columns(&["b"]).unwrap().num_rows(3);
    let df = DataFrame::read_parquet_with_options(opts).unwrap();

    assert_eq!(df.num_columns(), 1);
    assert_eq!(df.num_rows(), 3);
    assert_eq!(df.column_names(), vec!["b"]);

    let col = df.series_view("b").unwrap();
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

    let col = df2.series_view("x").unwrap();
    assert_eq!(col.to_vec::<i32>().unwrap(), vec![100, 200]);

    let col = df2.series_view("y").unwrap();
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

    assert_eq!(df2.series_view("alpha").unwrap().to_vec::<i64>().unwrap(), vec![10, 20, 30]);
    assert_eq!(df2.series_view("beta").unwrap().to_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
    assert_eq!(df2.series_view("gamma").unwrap().to_vec::<i32>().unwrap(), vec![7, 8, 9]);
}
