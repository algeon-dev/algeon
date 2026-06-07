//! Tests for Parquet and CSV I/O.

use std::process::Command;

use cudf::{
    ErrorKind,
    core::{Column, Table},
    io::{csv, parquet},
};
use tempfile::TempDir;

/// Write a Parquet file with the specified row group size using PyArrow.
///
/// Returns `true` if the file was written successfully, `false` if Python/PyArrow
/// is not available. Panics on unexpected errors.
fn write_parquet_with_row_groups(path: &std::path::Path, values: &[i32], row_group_size: usize) -> bool {
    let vals_str: Vec<String> = values.iter().map(|v| v.to_string()).collect();
    let script = format!(
        "import pyarrow as pa, pyarrow.parquet as pq; \
         t = pa.table({{'val': pa.array([{}], type=pa.int32())}}); \
         pq.write_table(t, '{}', row_group_size={})",
        vals_str.join(","),
        path.display(),
        row_group_size,
    );
    let output = Command::new("python3").args(["-c", &script]).output();
    match output {
        Ok(out) if out.status.success() => true,
        Ok(out) => {
            eprintln!("PyArrow write failed: {}", String::from_utf8_lossy(&out.stderr));
            false
        },
        Err(_) => false,
    }
}

#[test]
fn test_parquet_roundtrip() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.parquet");

    // Write
    let c1 = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let c2 = Column::from_slice(&[10.0f64, 20.0, 30.0]).unwrap();
    let table = Table::from_columns(vec![c1, c2]).unwrap();
    let view = table.view().unwrap();

    let write_opts =
        parquet::ParquetTableWriteOptions::builder().filepath(&path).column_names(&["ids", "values"]).build();
    parquet::write_parquet(&view, &write_opts).unwrap();

    // Read back
    let read_opts = parquet::ParquetTableReadOptions::builder().filepath(&path).build();
    let result = parquet::read_parquet(read_opts).unwrap();
    let read_view = result.table.view().unwrap();

    assert_eq!(read_view.num_columns().unwrap(), 2);
    assert_eq!(read_view.num_rows().unwrap(), 3);
    assert_eq!(result.column_names, Some(vec!["ids".to_string(), "values".to_string()]));

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

    let write_opts =
        parquet::ParquetTableWriteOptions::builder().filepath(&path).column_names(&["a", "b", "c"]).build();
    parquet::write_parquet(&view, &write_opts).unwrap();

    // Read only column "b"
    let read_opts = parquet::ParquetTableReadOptions::builder().filepath(&path).columns(&["b"]).build();
    let result = parquet::read_parquet(read_opts).unwrap();
    let read_view = result.table.view().unwrap();

    assert_eq!(read_view.num_columns().unwrap(), 1);
    assert_eq!(result.column_names, Some(vec!["b".to_string()]));
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

    let write_opts = parquet::ParquetTableWriteOptions::builder().filepath(&path).column_names(&["x"]).build();
    parquet::write_parquet(&view, &write_opts).unwrap();

    // Read only first 10 rows
    let read_opts = parquet::ParquetTableReadOptions::builder().filepath(&path).num_rows(10).build();
    let result = parquet::read_parquet(read_opts).unwrap();
    let read_view = result.table.view().unwrap();

    assert_eq!(read_view.num_rows().unwrap(), 10);
    assert_eq!(result.column_names, Some(vec!["x".to_string()]));
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

    let write_opts = csv::CsvTableWriteOptions::builder()
        .filepath(&path)
        .column_names(&["ids", "values"])
        .include_header(true)
        .build();
    csv::write_csv(&view, &write_opts).unwrap();

    // Read back
    let read_opts = csv::CsvTableReadOptions::builder().filepath(&path).has_header(true).build();
    let result = csv::read_csv(&read_opts).unwrap();
    let read_view = result.table.view().unwrap();

    assert_eq!(read_view.num_columns().unwrap(), 2);
    assert_eq!(read_view.num_rows().unwrap(), 3);
    assert_eq!(result.column_names, Some(vec!["ids".to_string(), "values".to_string()]));
}

#[test]
fn test_csv_read_without_header_returns_no_schema_metadata() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test_no_header.csv");

    std::fs::write(&path, "1,10\n2,20\n").unwrap();

    let read_opts = csv::CsvTableReadOptions::builder().filepath(&path).has_header(false).build();
    let result = csv::read_csv(&read_opts).unwrap();

    assert_eq!(result.column_names, None);
    assert_eq!(result.table.view().unwrap().num_rows().unwrap(), 2);
}

#[test]
fn test_parquet_read_returns_column_names_in_shared_result() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test_meta.parquet");

    // Write a parquet file with known column names
    let c1 = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let c2 = Column::from_slice(&[10.0f64, 20.0, 30.0]).unwrap();
    let c3 = Column::from_slice(&[100i64, 200, 300]).unwrap();
    let table = Table::from_columns(vec![c1, c2, c3]).unwrap();
    let view = table.view().unwrap();

    let write_opts =
        parquet::ParquetTableWriteOptions::builder().filepath(&path).column_names(&["alpha", "beta", "gamma"]).build();
    parquet::write_parquet(&view, &write_opts).unwrap();

    // Read back with metadata
    let read_opts = parquet::ParquetTableReadOptions::builder().filepath(&path).build();
    let result = parquet::read_parquet(read_opts).unwrap();

    // Verify column names
    assert_eq!(result.column_names, Some(vec!["alpha".to_string(), "beta".to_string(), "gamma".to_string()]));

    // Verify table data is intact
    let read_view = result.table.view().unwrap();
    assert_eq!(read_view.num_columns().unwrap(), 3);
    assert_eq!(read_view.num_rows().unwrap(), 3);

    let col0 = read_view.column(0).unwrap();
    assert_eq!(col0.to_vec::<i32>().unwrap(), vec![1, 2, 3]);

    let col1 = read_view.column(1).unwrap();
    assert_eq!(col1.to_vec::<f64>().unwrap(), vec![10.0, 20.0, 30.0]);

    let col2 = read_view.column(2).unwrap();
    assert_eq!(col2.to_vec::<i64>().unwrap(), vec![100, 200, 300]);
}

#[test]
fn test_parquet_read_column_selection_preserves_projected_names() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test_meta_select.parquet");

    let c1 = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let c2 = Column::from_slice(&[4i32, 5, 6]).unwrap();
    let c3 = Column::from_slice(&[7i32, 8, 9]).unwrap();
    let table = Table::from_columns(vec![c1, c2, c3]).unwrap();
    let view = table.view().unwrap();

    let write_opts =
        parquet::ParquetTableWriteOptions::builder().filepath(&path).column_names(&["x", "y", "z"]).build();
    parquet::write_parquet(&view, &write_opts).unwrap();

    // Read only columns "x" and "z"
    let read_opts = parquet::ParquetTableReadOptions::builder().filepath(&path).columns(&["x", "z"]).build();
    let result = parquet::read_parquet(read_opts).unwrap();

    // Only the selected columns should appear
    assert_eq!(result.column_names, Some(vec!["x".to_string(), "z".to_string()]));
    let read_view = result.table.view().unwrap();
    assert_eq!(read_view.num_columns().unwrap(), 2);

    let col0 = read_view.column(0).unwrap();
    assert_eq!(col0.to_vec::<i32>().unwrap(), vec![1, 2, 3]);

    let col1 = read_view.column(1).unwrap();
    assert_eq!(col1.to_vec::<i32>().unwrap(), vec![7, 8, 9]);
}

#[test]
fn test_parquet_read_single_column_preserves_name() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test_meta_single.parquet");

    let c1 = Column::from_slice(&[42i32]).unwrap();
    let table = Table::from_columns(vec![c1]).unwrap();
    let view = table.view().unwrap();

    let write_opts = parquet::ParquetTableWriteOptions::builder().filepath(&path).column_names(&["only_col"]).build();
    parquet::write_parquet(&view, &write_opts).unwrap();

    let read_opts = parquet::ParquetTableReadOptions::builder().filepath(&path).build();
    let result = parquet::read_parquet(read_opts).unwrap();

    assert_eq!(result.column_names, Some(vec!["only_col".to_string()]));
    assert_eq!(result.table.view().unwrap().num_columns().unwrap(), 1);
}

#[test]
fn test_parquet_row_group_selection() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test_rg.parquet");

    // Write 9 rows with 3 rows per row group → 3 groups (via PyArrow)
    let values: Vec<i32> = vec![10, 20, 30, 40, 50, 60, 70, 80, 90];
    if !write_parquet_with_row_groups(&path, &values, 3) {
        eprintln!("Skipping: python3/pyarrow not available");
        return;
    }

    // Read only row group 1 (rows 40, 50, 60)
    let read_opts = parquet::ParquetTableReadOptions::builder().filepath(&path).row_groups(&[1]).build();
    let result = parquet::read_parquet(read_opts).unwrap();
    let read_view = result.table.view().unwrap();

    assert_eq!(read_view.num_rows().unwrap(), 3);
    assert_eq!(result.column_names, Some(vec!["val".to_string()]));
    let vals = read_view.column(0).unwrap().to_vec::<i32>().unwrap();
    assert_eq!(vals, vec![40, 50, 60]);
}

#[test]
fn test_parquet_row_group_multiple_selection() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test_rg_multi.parquet");

    // Write 9 rows with 3 rows per row group → 3 groups (via PyArrow)
    let values: Vec<i32> = vec![10, 20, 30, 40, 50, 60, 70, 80, 90];
    if !write_parquet_with_row_groups(&path, &values, 3) {
        eprintln!("Skipping: python3/pyarrow not available");
        return;
    }

    // Read row groups 0 and 2 (rows 10,20,30 and 70,80,90)
    let read_opts = parquet::ParquetTableReadOptions::builder().filepath(&path).row_groups(&[0, 2]).build();
    let result = parquet::read_parquet(read_opts).unwrap();
    let read_view = result.table.view().unwrap();

    assert_eq!(read_view.num_rows().unwrap(), 6);
    assert_eq!(result.column_names, Some(vec!["val".to_string()]));
    let vals = read_view.column(0).unwrap().to_vec::<i32>().unwrap();
    assert_eq!(vals, vec![10, 20, 30, 70, 80, 90]);
}

#[test]
fn test_parquet_row_group_empty_selection() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test_rg_empty.parquet");

    let col = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let table = Table::from_columns(vec![col]).unwrap();
    let view = table.view().unwrap();

    let write_opts = parquet::ParquetTableWriteOptions::builder().filepath(&path).column_names(&["x"]).build();
    parquet::write_parquet(&view, &write_opts).unwrap();

    // Empty row_groups slice — should read all rows (libcudf treats empty as "all")
    let read_opts = parquet::ParquetTableReadOptions::builder().filepath(&path).row_groups(&[]).build();
    let result = parquet::read_parquet(read_opts).unwrap();
    let read_view = result.table.view().unwrap();

    assert_eq!(read_view.num_rows().unwrap(), 3);
    assert_eq!(result.column_names, Some(vec!["x".to_string()]));
    let vals = read_view.column(0).unwrap().to_vec::<i32>().unwrap();
    assert_eq!(vals, vec![1, 2, 3]);
}

#[test]
fn test_parquet_row_groups_with_skip_rows_errors() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test_rg_skip_err.parquet");

    let col = Column::from_slice(&[1i32, 2, 3, 4, 5, 6]).unwrap();
    let table = Table::from_columns(vec![col]).unwrap();
    let view = table.view().unwrap();

    let write_opts = parquet::ParquetTableWriteOptions::builder().filepath(&path).column_names(&["x"]).build();
    parquet::write_parquet(&view, &write_opts).unwrap();

    // Combining row_groups with skip_rows should error (Rust-side validation)
    let read_opts = parquet::ParquetTableReadOptions::builder().filepath(&path).row_groups(&[0]).skip_rows(1).build();
    let result = parquet::read_parquet(read_opts);
    let err = result.err().expect("expected error when combining row_groups with skip_rows");
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
}

#[test]
fn test_parquet_row_groups_with_num_rows_errors() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test_rg_num_err.parquet");

    let col = Column::from_slice(&[1i32, 2, 3, 4, 5, 6]).unwrap();
    let table = Table::from_columns(vec![col]).unwrap();
    let view = table.view().unwrap();

    let write_opts = parquet::ParquetTableWriteOptions::builder().filepath(&path).column_names(&["x"]).build();
    parquet::write_parquet(&view, &write_opts).unwrap();

    // Combining row_groups with num_rows should error (Rust-side validation)
    let read_opts = parquet::ParquetTableReadOptions::builder().filepath(&path).row_groups(&[0]).num_rows(2).build();
    let result = parquet::read_parquet(read_opts);
    let err = result.err().expect("expected error when combining row_groups with num_rows");
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
}

#[test]
fn test_csv_custom_delimiter() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test_tab.csv");

    let c1 = Column::from_slice(&[1i32, 2]).unwrap();
    let table = Table::from_columns(vec![c1]).unwrap();
    let view = table.view().unwrap();

    let write_opts = csv::CsvTableWriteOptions::builder().filepath(&path).column_names(&["x"]).delimiter(b'\t').build();
    csv::write_csv(&view, &write_opts).unwrap();

    let read_opts = csv::CsvTableReadOptions::builder().filepath(&path).delimiter(b'\t').build();
    let result = csv::read_csv(&read_opts).unwrap();
    assert_eq!(result.column_names, Some(vec!["x".to_string()]));
    assert_eq!(result.table.view().unwrap().num_rows().unwrap(), 2);
}
