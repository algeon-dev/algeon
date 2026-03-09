//! Tests for Parquet and CSV I/O.

use cudf::column::Column;
use cudf::io::{csv, parquet};
use cudf::table::Table;

use tempfile::TempDir;

#[test]
fn test_parquet_roundtrip() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.parquet");

    // Write
    let c1 = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let c2 = Column::from_slice(&[10.0f64, 20.0, 30.0]).unwrap();
    let table = Table::from_columns(vec![c1, c2]).unwrap();
    let view = table.view().unwrap();

    let write_opts = parquet::ParquetWriterOptions::new(&path)
        .unwrap()
        .column_names(&["ids", "values"]);
    parquet::write_parquet(&view, &write_opts).unwrap();

    // Read back
    let read_opts = parquet::ParquetReaderOptions::new(&path).unwrap();
    let read_table = parquet::read_parquet(&read_opts).unwrap();
    let read_view = read_table.view().unwrap();

    assert_eq!(read_view.num_columns().unwrap(), 2);
    assert_eq!(read_view.num_rows().unwrap(), 3);

    let col0 = read_view.column(0).unwrap();
    assert_eq!(col0.to_vec::<i32>().unwrap(), vec![1, 2, 3]);

    let col1 = read_view.column(1).unwrap();
    assert_eq!(col1.to_vec::<f64>().unwrap(), vec![10.0, 20.0, 30.0]);
}

#[test]
fn test_parquet_column_selection() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test_select.parquet");

    let c1 = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let c2 = Column::from_slice(&[4i32, 5, 6]).unwrap();
    let c3 = Column::from_slice(&[7i32, 8, 9]).unwrap();
    let table = Table::from_columns(vec![c1, c2, c3]).unwrap();
    let view = table.view().unwrap();

    let write_opts = parquet::ParquetWriterOptions::new(&path)
        .unwrap()
        .column_names(&["a", "b", "c"]);
    parquet::write_parquet(&view, &write_opts).unwrap();

    // Read only column "b"
    let read_opts = parquet::ParquetReaderOptions::new(&path)
        .unwrap()
        .columns(&["b"]);
    let read_table = parquet::read_parquet(&read_opts).unwrap();
    let read_view = read_table.view().unwrap();

    assert_eq!(read_view.num_columns().unwrap(), 1);
    let col = read_view.column(0).unwrap();
    assert_eq!(col.to_vec::<i32>().unwrap(), vec![4, 5, 6]);
}

#[test]
fn test_parquet_row_limits() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test_rows.parquet");

    let data: Vec<i32> = (0..100).collect();
    let col = Column::from_slice(&data).unwrap();
    let table = Table::from_columns(vec![col]).unwrap();
    let view = table.view().unwrap();

    let write_opts = parquet::ParquetWriterOptions::new(&path)
        .unwrap()
        .column_names(&["x"]);
    parquet::write_parquet(&view, &write_opts).unwrap();

    // Read only first 10 rows
    let read_opts = parquet::ParquetReaderOptions::new(&path)
        .unwrap()
        .num_rows(10);
    let read_table = parquet::read_parquet(&read_opts).unwrap();
    let read_view = read_table.view().unwrap();

    assert_eq!(read_view.num_rows().unwrap(), 10);
    let col = read_view.column(0).unwrap();
    let vals = col.to_vec::<i32>().unwrap();
    assert_eq!(vals, (0..10).collect::<Vec<i32>>());
}

#[test]
fn test_csv_roundtrip() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.csv");

    // Write
    let c1 = Column::from_slice(&[1i64, 2, 3]).unwrap();
    let c2 = Column::from_slice(&[10.0f64, 20.0, 30.0]).unwrap();
    let table = Table::from_columns(vec![c1, c2]).unwrap();
    let view = table.view().unwrap();

    let write_opts = csv::CsvWriterOptions::new(&path)
        .unwrap()
        .column_names(&["ids", "values"])
        .include_header(true);
    csv::write_csv(&view, &write_opts).unwrap();

    // Read back
    let read_opts = csv::CsvReaderOptions::new(&path)
        .unwrap()
        .has_header(true);
    let read_table = csv::read_csv(&read_opts).unwrap();
    let read_view = read_table.view().unwrap();

    assert_eq!(read_view.num_columns().unwrap(), 2);
    assert_eq!(read_view.num_rows().unwrap(), 3);
}

#[test]
fn test_csv_custom_delimiter() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test_tab.csv");

    let c1 = Column::from_slice(&[1i32, 2]).unwrap();
    let table = Table::from_columns(vec![c1]).unwrap();
    let view = table.view().unwrap();

    let write_opts = csv::CsvWriterOptions::new(&path)
        .unwrap()
        .column_names(&["x"])
        .delimiter(b'\t');
    csv::write_csv(&view, &write_opts).unwrap();

    let read_opts = csv::CsvReaderOptions::new(&path)
        .unwrap()
        .delimiter(b'\t');
    let read_table = csv::read_csv(&read_opts).unwrap();
    assert_eq!(read_table.view().unwrap().num_rows().unwrap(), 2);
}
