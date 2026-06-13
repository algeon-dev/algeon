//! Tests for Parquet and CSV I/O.
#[cfg(unix)]
use std::os::unix::ffi::OsStringExt;
use std::process::Command;

use cudf::{
    ErrorCode, ErrorKind,
    core::{
        Table,
        error::{FactKeyName, FactValue},
    },
    io::{csv, parquet},
    testing::prelude::*,
};
use tempfile::TempDir;

use crate::common::{
    arm_next_io_metadata_fault, column_from_slice, error_context_value, reset_io_metadata_fault, sample_runtime,
};

struct MetadataFaultReset;

impl Drop for MetadataFaultReset {
    fn drop(&mut self) {
        reset_io_metadata_fault();
    }
}

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
    let rt = sample_runtime();

    // Write
    let c1 = column_from_slice(&[1i32, 2, 3]).unwrap();
    let c2 = column_from_slice(&[10.0f64, 20.0, 30.0]).unwrap();
    let table = Table::from_columns(vec![c1, c2]).unwrap();

    let write_opts =
        parquet::ParquetTableWriteOptions::builder().filepath(&path).column_names(&["ids", "values"]).build();
    parquet::write_parquet(&table, &write_opts).unwrap();

    // Read back
    let read_opts = parquet::ParquetTableReadOptions::builder().filepath(&path).with_runtime(&rt).build();
    let result = parquet::read_parquet_table(&read_opts).unwrap();
    let read_view = result.table.testing_view().unwrap();

    assert_eq!(read_view.num_columns().unwrap(), 2);
    assert_eq!(read_view.num_rows().unwrap(), 3);
    assert_eq!(result.column_names, Some(vec!["ids".to_string(), "values".to_string()]));
    assert_eq!(result.table.testing_execution_domain_id(), rt.testing_execution_domain_id());
    assert_eq!(result.table.testing_execution_domain_stream_value(), rt.stream().testing_raw_handle_value());

    let col0 = read_view.column(0).unwrap();
    assert_eq!(col0.to_vec::<i32>().unwrap(), vec![1, 2, 3]);

    let col1 = read_view.column(1).unwrap();
    assert_eq!(col1.to_vec::<f64>().unwrap(), vec![10.0, 20.0, 30.0]);
}

#[test]
fn test_parquet_column_selection() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test_select.parquet");
    let rt = sample_runtime();

    let c1 = column_from_slice(&[1i32, 2, 3]).unwrap();
    let c2 = column_from_slice(&[4i32, 5, 6]).unwrap();
    let c3 = column_from_slice(&[7i32, 8, 9]).unwrap();
    let table = Table::from_columns(vec![c1, c2, c3]).unwrap();

    let write_opts =
        parquet::ParquetTableWriteOptions::builder().filepath(&path).column_names(&["a", "b", "c"]).build();
    parquet::write_parquet(&table, &write_opts).unwrap();

    // Read only column "b"
    let read_opts =
        parquet::ParquetTableReadOptions::builder().filepath(&path).columns(&["b"]).with_runtime(&rt).build();
    let result = parquet::read_parquet_table(&read_opts).unwrap();
    let read_view = result.table.testing_view().unwrap();

    assert_eq!(read_view.num_columns().unwrap(), 1);
    assert_eq!(result.column_names, Some(vec!["b".to_string()]));
    let col = read_view.column(0).unwrap();
    assert_eq!(col.to_vec::<i32>().unwrap(), vec![4, 5, 6]);
}

#[test]
fn test_parquet_row_limits() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test_rows.parquet");
    let rt = sample_runtime();

    let data: Vec<i32> = (0..100).collect();
    let col = column_from_slice(&data).unwrap();
    let table = Table::from_columns(vec![col]).unwrap();

    let write_opts = parquet::ParquetTableWriteOptions::builder().filepath(&path).column_names(&["x"]).build();
    parquet::write_parquet(&table, &write_opts).unwrap();

    // Read only first 10 rows
    let read_opts = parquet::ParquetTableReadOptions::builder().filepath(&path).num_rows(10).with_runtime(&rt).build();
    let result = parquet::read_parquet_table(&read_opts).unwrap();
    let read_view = result.table.testing_view().unwrap();

    assert_eq!(read_view.num_rows().unwrap(), 10);
    assert_eq!(result.column_names, Some(vec!["x".to_string()]));
    let col = read_view.column(0).unwrap();
    let vals = col.to_vec::<i32>().unwrap();
    assert_eq!(vals, (0..10).collect::<Vec<i32>>());
}

#[test]
fn test_parquet_skip_rows_and_row_limits() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test_skip_rows.parquet");
    let rt = sample_runtime();

    let data: Vec<i32> = (0..20).collect();
    let col = column_from_slice(&data).unwrap();
    let table = Table::from_columns(vec![col]).unwrap();

    let write_opts = parquet::ParquetTableWriteOptions::builder().filepath(&path).column_names(&["x"]).build();
    parquet::write_parquet(&table, &write_opts).unwrap();

    let read_opts =
        parquet::ParquetTableReadOptions::builder().filepath(&path).skip_rows(5).num_rows(4).with_runtime(&rt).build();
    let result = parquet::read_parquet_table(&read_opts).unwrap();
    let read_view = result.table.testing_view().unwrap();

    assert_eq!(read_view.num_rows().unwrap(), 4);
    let col = read_view.column(0).unwrap();
    assert_eq!(col.to_vec::<i32>().unwrap(), vec![5, 6, 7, 8]);
}

#[test]
fn test_csv_roundtrip() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.csv");
    let rt = sample_runtime();

    // Write
    let c1 = column_from_slice(&[1i64, 2, 3]).unwrap();
    let c2 = column_from_slice(&[10.0f64, 20.0, 30.0]).unwrap();
    let table = Table::from_columns(vec![c1, c2]).unwrap();

    let write_opts = csv::CsvTableWriteOptions::builder()
        .filepath(&path)
        .column_names(&["ids", "values"])
        .include_header(true)
        .build();
    csv::write_csv(&table, &write_opts).unwrap();

    // Read back
    let read_opts = csv::CsvTableReadOptions::builder().filepath(&path).has_header(true).with_runtime(&rt).build();
    let result = csv::read_csv(&read_opts).unwrap();
    let read_view = result.table.testing_view().unwrap();

    assert_eq!(read_view.num_columns().unwrap(), 2);
    assert_eq!(read_view.num_rows().unwrap(), 3);
    assert_eq!(result.column_names, Some(vec!["ids".to_string(), "values".to_string()]));
    assert_eq!(result.table.testing_execution_domain_id(), rt.testing_execution_domain_id());
    assert_eq!(result.table.testing_execution_domain_stream_value(), rt.stream().testing_raw_handle_value());
}

#[test]
fn test_csv_read_without_header_returns_no_schema_metadata() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test_no_header.csv");
    let rt = sample_runtime();

    std::fs::write(
        &path,
        "1,10
2,20
",
    )
    .unwrap();

    let read_opts = csv::CsvTableReadOptions::builder().filepath(&path).has_header(false).with_runtime(&rt).build();
    let result = csv::read_csv(&read_opts).unwrap();

    assert_eq!(result.column_names, None);
    assert_eq!(result.table.testing_view().unwrap().num_rows().unwrap(), 2);
}

#[test]
fn test_parquet_read_returns_column_names_in_shared_result() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test_meta.parquet");
    let rt = sample_runtime();

    // Write a parquet file with known column names
    let c1 = column_from_slice(&[1i32, 2, 3]).unwrap();
    let c2 = column_from_slice(&[10.0f64, 20.0, 30.0]).unwrap();
    let c3 = column_from_slice(&[100i64, 200, 300]).unwrap();
    let table = Table::from_columns(vec![c1, c2, c3]).unwrap();

    let write_opts =
        parquet::ParquetTableWriteOptions::builder().filepath(&path).column_names(&["alpha", "beta", "gamma"]).build();
    parquet::write_parquet(&table, &write_opts).unwrap();

    // Read back with metadata
    let read_opts = parquet::ParquetTableReadOptions::builder().filepath(&path).with_runtime(&rt).build();
    let result = parquet::read_parquet_table(&read_opts).unwrap();

    // Verify column names
    assert_eq!(result.column_names, Some(vec!["alpha".to_string(), "beta".to_string(), "gamma".to_string()]));

    // Verify table data is intact
    let read_view = result.table.testing_view().unwrap();
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
    let rt = sample_runtime();

    let c1 = column_from_slice(&[1i32, 2, 3]).unwrap();
    let c2 = column_from_slice(&[4i32, 5, 6]).unwrap();
    let c3 = column_from_slice(&[7i32, 8, 9]).unwrap();
    let table = Table::from_columns(vec![c1, c2, c3]).unwrap();

    let write_opts =
        parquet::ParquetTableWriteOptions::builder().filepath(&path).column_names(&["x", "y", "z"]).build();
    parquet::write_parquet(&table, &write_opts).unwrap();

    // Read only columns "x" and "z"
    let read_opts =
        parquet::ParquetTableReadOptions::builder().filepath(&path).columns(&["x", "z"]).with_runtime(&rt).build();
    let result = parquet::read_parquet_table(&read_opts).unwrap();

    // Only the selected columns should appear
    assert_eq!(result.column_names, Some(vec!["x".to_string(), "z".to_string()]));
    let read_view = result.table.testing_view().unwrap();
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
    let rt = sample_runtime();

    let c1 = column_from_slice(&[42i32]).unwrap();
    let table = Table::from_columns(vec![c1]).unwrap();

    let write_opts = parquet::ParquetTableWriteOptions::builder().filepath(&path).column_names(&["only_col"]).build();
    parquet::write_parquet(&table, &write_opts).unwrap();

    let read_opts = parquet::ParquetTableReadOptions::builder().filepath(&path).with_runtime(&rt).build();
    let result = parquet::read_parquet_table(&read_opts).unwrap();

    assert_eq!(result.column_names, Some(vec!["only_col".to_string()]));
    assert_eq!(result.table.testing_view().unwrap().num_columns().unwrap(), 1);
}

#[test]
fn test_parquet_row_group_selection() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test_rg.parquet");
    let rt = sample_runtime();

    // Write 9 rows with 3 rows per row group → 3 groups (via PyArrow)
    let values: Vec<i32> = vec![10, 20, 30, 40, 50, 60, 70, 80, 90];
    if !write_parquet_with_row_groups(&path, &values, 3) {
        eprintln!("Skipping: python3/pyarrow not available");
        return;
    }

    // Read only row group 1 (rows 40, 50, 60)
    let read_opts =
        parquet::ParquetTableReadOptions::builder().filepath(&path).row_groups(&[1]).with_runtime(&rt).build();
    let result = parquet::read_parquet_table(&read_opts).unwrap();
    let read_view = result.table.testing_view().unwrap();

    assert_eq!(read_view.num_rows().unwrap(), 3);
    assert_eq!(result.column_names, Some(vec!["val".to_string()]));
    let vals = read_view.column(0).unwrap().to_vec::<i32>().unwrap();
    assert_eq!(vals, vec![40, 50, 60]);
}

#[test]
fn test_parquet_row_group_multiple_selection() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test_rg_multi.parquet");
    let rt = sample_runtime();

    // Write 9 rows with 3 rows per row group → 3 groups (via PyArrow)
    let values: Vec<i32> = vec![10, 20, 30, 40, 50, 60, 70, 80, 90];
    if !write_parquet_with_row_groups(&path, &values, 3) {
        eprintln!("Skipping: python3/pyarrow not available");
        return;
    }

    // Read row groups 0 and 2 (rows 10,20,30 and 70,80,90)
    let read_opts =
        parquet::ParquetTableReadOptions::builder().filepath(&path).row_groups(&[0, 2]).with_runtime(&rt).build();
    let result = parquet::read_parquet_table(&read_opts).unwrap();
    let read_view = result.table.testing_view().unwrap();

    assert_eq!(read_view.num_rows().unwrap(), 6);
    assert_eq!(result.column_names, Some(vec!["val".to_string()]));
    let vals = read_view.column(0).unwrap().to_vec::<i32>().unwrap();
    assert_eq!(vals, vec![10, 20, 30, 70, 80, 90]);
}

#[test]
fn test_parquet_row_group_empty_selection() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test_rg_empty.parquet");
    let rt = sample_runtime();

    let col = column_from_slice(&[1i32, 2, 3]).unwrap();
    let table = Table::from_columns(vec![col]).unwrap();

    let write_opts = parquet::ParquetTableWriteOptions::builder().filepath(&path).column_names(&["x"]).build();
    parquet::write_parquet(&table, &write_opts).unwrap();

    // Empty row_groups slice — should read all rows (libcudf treats empty as "all")
    let read_opts =
        parquet::ParquetTableReadOptions::builder().filepath(&path).row_groups(&[]).with_runtime(&rt).build();
    let result = parquet::read_parquet_table(&read_opts).unwrap();
    let read_view = result.table.testing_view().unwrap();

    assert_eq!(read_view.num_rows().unwrap(), 3);
    assert_eq!(result.column_names, Some(vec!["x".to_string()]));
    let vals = read_view.column(0).unwrap().to_vec::<i32>().unwrap();
    assert_eq!(vals, vec![1, 2, 3]);
}

#[test]
fn test_parquet_row_groups_with_skip_rows_errors() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test_rg_skip_err.parquet");
    let rt = sample_runtime();

    let col = column_from_slice(&[1i32, 2, 3, 4, 5, 6]).unwrap();
    let table = Table::from_columns(vec![col]).unwrap();

    let write_opts = parquet::ParquetTableWriteOptions::builder().filepath(&path).column_names(&["x"]).build();
    parquet::write_parquet(&table, &write_opts).unwrap();

    // Combining row_groups with skip_rows should error (Rust-side validation)
    let read_opts = parquet::ParquetTableReadOptions::builder()
        .filepath(&path)
        .row_groups(&[0])
        .skip_rows(1)
        .with_runtime(&rt)
        .build();
    let err = match parquet::read_parquet_table(&read_opts) {
        Ok(_) => panic!("expected error when combining row_groups with skip_rows"),
        Err(err) => err,
    };
    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(err.operation(), "io::parquet::read_parquet_table");
    assert_eq!(error_context_value(&err, "row_group_count").as_deref(), Some("1"));
    assert_eq!(error_context_value(&err, "skip_rows").as_deref(), Some("1"));
    assert_eq!(error_context_value(&err, "num_rows").as_deref(), Some("all"));
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("row_groups"));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("incompatible_option_combination"));
    assert_eq!(
        error_context_value(&err, "remedy").as_deref(),
        Some("use row_groups by itself, or remove row_groups and use skip_rows/num_rows")
    );
}

#[test]
fn test_parquet_row_groups_with_num_rows_errors() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test_rg_num_err.parquet");
    let rt = sample_runtime();

    let col = column_from_slice(&[1i32, 2, 3, 4, 5, 6]).unwrap();
    let table = Table::from_columns(vec![col]).unwrap();

    let write_opts = parquet::ParquetTableWriteOptions::builder().filepath(&path).column_names(&["x"]).build();
    parquet::write_parquet(&table, &write_opts).unwrap();

    // Combining row_groups with num_rows should error (Rust-side validation)
    let read_opts = parquet::ParquetTableReadOptions::builder()
        .filepath(&path)
        .row_groups(&[0])
        .num_rows(2)
        .with_runtime(&rt)
        .build();
    let err = match parquet::read_parquet_table(&read_opts) {
        Ok(_) => panic!("expected error when combining row_groups with num_rows"),
        Err(err) => err,
    };
    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(err.operation(), "io::parquet::read_parquet_table");
    assert_eq!(error_context_value(&err, "row_group_count").as_deref(), Some("1"));
    assert_eq!(error_context_value(&err, "skip_rows").as_deref(), Some("0"));
    assert_eq!(error_context_value(&err, "num_rows").as_deref(), Some("2"));
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("row_groups"));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("incompatible_option_combination"));
    assert_eq!(
        error_context_value(&err, "remedy").as_deref(),
        Some("use row_groups by itself, or remove row_groups and use skip_rows/num_rows")
    );
}

#[test]
fn test_csv_write_rejects_column_name_with_nul_byte_using_stable_contract() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("bad_header.csv");
    let table = Table::from_columns(vec![column_from_slice(&[1i32]).unwrap()]).unwrap();

    let write_opts = csv::CsvTableWriteOptions::builder().filepath(&path).column_names(&["bad\0name"]).build();
    let err = csv::write_csv(&table, &write_opts).expect_err("column names containing NUL should fail before FFI");

    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.code(), ErrorCode::NullUnexpected);
    assert_eq!(err.operation(), "io::csv::write_csv");
    assert_eq!(error_context_value(&err, "column_name").as_deref(), Some("bad\0name"));
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("column_name"));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("interior_null_byte"));
}

#[test]
fn test_parquet_write_rejects_column_name_with_nul_byte_using_stable_contract() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("bad_header.parquet");
    let table = Table::from_columns(vec![column_from_slice(&[1i32]).unwrap()]).unwrap();

    let write_opts = parquet::ParquetTableWriteOptions::builder().filepath(&path).column_names(&["bad\0name"]).build();
    let err =
        parquet::write_parquet(&table, &write_opts).expect_err("column names containing NUL should fail before FFI");

    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.code(), ErrorCode::NullUnexpected);
    assert_eq!(err.operation(), "io::parquet::write_parquet");
    assert_eq!(error_context_value(&err, "column_name").as_deref(), Some("bad\0name"));
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("column_name"));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("interior_null_byte"));
}

#[test]
fn test_parquet_read_rejects_projected_column_name_with_nul_byte_using_stable_contract() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("bad_projection.parquet");
    std::fs::write(&path, []).unwrap();
    let rt = sample_runtime();

    let read_opts =
        parquet::ParquetTableReadOptions::builder().filepath(&path).columns(&["bad\0name"]).with_runtime(&rt).build();
    let err = match parquet::read_parquet_table(&read_opts) {
        Ok(_) => panic!("projected column names containing NUL should fail before FFI"),
        Err(err) => err,
    };

    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.code(), ErrorCode::NullUnexpected);
    assert_eq!(err.operation(), "io::parquet::read_parquet_table");
    assert_eq!(error_context_value(&err, "column_name").as_deref(), Some("bad\0name"));
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("column_name"));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("interior_null_byte"));
}

#[test]
fn test_parquet_skip_rows_exceeding_i64_max_uses_stable_overflow_contract() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("overflow_skip_rows.parquet");
    std::fs::write(&path, []).unwrap();
    let rt = sample_runtime();

    let read_opts = parquet::ParquetTableReadOptions::builder()
        .filepath(&path)
        .skip_rows((i64::MAX as usize) + 1)
        .with_runtime(&rt)
        .build();
    let err = match parquet::read_parquet_table(&read_opts) {
        Ok(_) => panic!("skip_rows overflow should fail before FFI"),
        Err(err) => err,
    };

    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(err.operation(), "io::parquet::read_parquet_table");
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("skip_rows"));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("exceeds_i64_max"));
    assert_eq!(error_context_value(&err, "remedy").as_deref(), Some("reduce skip_rows to fit within i64::MAX"));
    let expected = ((i64::MAX as usize) + 1).to_string();
    assert_eq!(error_context_value(&err, "skip_rows").as_deref(), Some(expected.as_str()));
}

#[test]
fn test_parquet_num_rows_exceeding_i64_max_uses_stable_overflow_contract() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("overflow_num_rows.parquet");
    std::fs::write(&path, []).unwrap();
    let rt = sample_runtime();

    let read_opts = parquet::ParquetTableReadOptions::builder()
        .filepath(&path)
        .num_rows((i64::MAX as usize) + 1)
        .with_runtime(&rt)
        .build();
    let err = match parquet::read_parquet_table(&read_opts) {
        Ok(_) => panic!("num_rows overflow should fail before FFI"),
        Err(err) => err,
    };

    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(err.operation(), "io::parquet::read_parquet_table");
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("num_rows"));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("exceeds_i64_max"));
    assert_eq!(error_context_value(&err, "remedy").as_deref(), Some("reduce num_rows to fit within i64::MAX"));
    let expected = ((i64::MAX as usize) + 1).to_string();
    assert_eq!(error_context_value(&err, "num_rows").as_deref(), Some(expected.as_str()));
}

#[test]
fn test_csv_write_rejects_column_name_count_with_stable_contract() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("bad_names.csv");
    let table = Table::from_columns(vec![column_from_slice(&[1i32, 2, 3]).unwrap()]).unwrap();

    let write_opts = csv::CsvTableWriteOptions::builder().filepath(&path).column_names(&["only", "extra"]).build();
    let err = csv::write_csv(&table, &write_opts).expect_err("column name count mismatch should fail before FFI");

    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.code(), ErrorCode::ColumnLengthMismatch);
    assert_eq!(err.operation(), "io::csv::write_csv");
    assert_eq!(err.fact_value(FactKeyName::ActualCount), Some(&FactValue::Usize(2)));
    assert_eq!(err.fact_value(FactKeyName::TableColumnCount), Some(&FactValue::Usize(1)));
    assert_eq!(error_context_value(&err, "column_name_count").as_deref(), Some("2"));
    assert_eq!(error_context_value(&err, "table_column_count").as_deref(), Some("1"));
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("column_name_count"));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("mismatch_with_table_column_count"));
    assert_eq!(
        error_context_value(&err, "remedy").as_deref(),
        Some("ensure column_name_count matches table_column_count")
    );
}

#[test]
fn test_parquet_write_rejects_column_name_count_with_stable_contract() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("bad_names.parquet");
    let table = Table::from_columns(vec![column_from_slice(&[1i32, 2, 3]).unwrap()]).unwrap();

    let write_opts =
        parquet::ParquetTableWriteOptions::builder().filepath(&path).column_names(&["only", "extra"]).build();
    let err =
        parquet::write_parquet(&table, &write_opts).expect_err("column name count mismatch should fail before FFI");

    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.code(), ErrorCode::ColumnLengthMismatch);
    assert_eq!(err.operation(), "io::parquet::write_parquet");
    assert_eq!(err.fact_value(FactKeyName::ActualCount), Some(&FactValue::Usize(2)));
    assert_eq!(err.fact_value(FactKeyName::TableColumnCount), Some(&FactValue::Usize(1)));
    assert_eq!(error_context_value(&err, "column_name_count").as_deref(), Some("2"));
    assert_eq!(error_context_value(&err, "table_column_count").as_deref(), Some("1"));
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("column_name_count"));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("mismatch_with_table_column_count"));
    assert_eq!(
        error_context_value(&err, "remedy").as_deref(),
        Some("ensure column_name_count matches table_column_count")
    );
}

#[test]
fn test_parquet_write_io_failure_uses_canonical_column_name_count_context() {
    let path =
        std::env::temp_dir().join("cudf_rs_missing_parent_dir_for_parquet_write_12345").join("write_failure.parquet");
    let table = Table::from_columns(vec![column_from_slice(&[1i32, 2, 3]).unwrap()]).unwrap();

    let write_opts = parquet::ParquetTableWriteOptions::builder().filepath(&path).column_names(&["x"]).build();
    let err = parquet::write_parquet(&table, &write_opts).expect_err("missing parent directory should fail in FFI");

    assert_eq!(err.operation(), "io::parquet::write_parquet");
    assert_eq!(error_context_value(&err, "column_name_count").as_deref(), Some("1"));
    assert_eq!(error_context_value(&err, "output_column_name_count").as_deref(), None);
}

#[test]
fn test_csv_write_io_failure_uses_canonical_column_name_count_context() {
    let path = std::env::temp_dir().join("cudf_rs_missing_parent_dir_for_csv_write_12345").join("write_failure.csv");
    let table = Table::from_columns(vec![column_from_slice(&[1i32, 2, 3]).unwrap()]).unwrap();

    let write_opts = csv::CsvTableWriteOptions::builder().filepath(&path).column_names(&["x"]).build();
    let err = csv::write_csv(&table, &write_opts).expect_err("missing parent directory should fail in FFI");

    assert_eq!(err.operation(), "io::csv::write_csv");
    assert_eq!(error_context_value(&err, "column_name_count").as_deref(), Some("1"));
    assert_eq!(error_context_value(&err, "num_column_names").as_deref(), None);
}

#[test]
fn test_parquet_read_metadata_null_name_entry_maps_to_invariant_violation() {
    let _reset = MetadataFaultReset;
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("metadata_fault.parquet");
    let rt = sample_runtime();

    let table = Table::from_columns(vec![column_from_slice(&[1i32, 2, 3]).unwrap()]).unwrap();
    let write_opts = parquet::ParquetTableWriteOptions::builder().filepath(&path).column_names(&["x"]).build();
    parquet::write_parquet(&table, &write_opts).unwrap();

    arm_next_io_metadata_fault(cudf_sys::CUDF_TEST_IO_METADATA_FAULT_NULL_NAME_ENTRY);
    let err = match parquet::read_parquet_table(
        &parquet::ParquetTableReadOptions::builder().filepath(&path).with_runtime(&rt).build(),
    ) {
        Ok(_) => panic!("malformed metadata should fail"),
        Err(err) => err,
    };

    assert_eq!(err.kind(), ErrorKind::InvariantViolation);
    assert_eq!(err.code(), ErrorCode::ForeignInvariantViolation);
    assert_eq!(err.operation(), "io::parquet::read_parquet_table");
    assert_eq!(err.fact_value(FactKeyName::ColumnIndex), Some(&FactValue::Usize(0)));
    assert_eq!(error_context_value(&err, "column_index").as_deref(), Some("0"));
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("column_name"));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("null_name_entry"));
    assert_eq!(error_context_value(&err, "name_index").as_deref(), None);
}

#[test]
fn test_parquet_read_metadata_null_name_array_uses_canonical_fault_context() {
    let _reset = MetadataFaultReset;
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("metadata_fault_null_array.parquet");
    let rt = sample_runtime();

    let table = Table::from_columns(vec![column_from_slice(&[1i32, 2, 3]).unwrap()]).unwrap();
    let write_opts = parquet::ParquetTableWriteOptions::builder().filepath(&path).column_names(&["x"]).build();
    parquet::write_parquet(&table, &write_opts).unwrap();

    arm_next_io_metadata_fault(cudf_sys::CUDF_TEST_IO_METADATA_FAULT_NULL_NAME_ARRAY);
    let err = match parquet::read_parquet_table(
        &parquet::ParquetTableReadOptions::builder().filepath(&path).with_runtime(&rt).build(),
    ) {
        Ok(_) => panic!("malformed metadata should fail"),
        Err(err) => err,
    };

    assert_eq!(err.kind(), ErrorKind::InvariantViolation);
    assert_eq!(err.code(), ErrorCode::ForeignInvariantViolation);
    assert_eq!(err.operation(), "io::parquet::read_parquet_table");
    assert_eq!(error_context_value(&err, "column_name_count").as_deref(), Some("1"));
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("column_names"));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("null_name_array"));
    assert_eq!(
        error_context_value(&err, "remedy").as_deref(),
        Some("ensure the foreign reader returns a name array whenever column_name_count is positive")
    );
    assert_eq!(error_context_value(&err, "names_ptr_null").as_deref(), None);
}

#[test]
fn test_parquet_read_metadata_invalid_utf8_name_uses_canonical_fault_context() {
    let _reset = MetadataFaultReset;
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("metadata_fault_invalid_utf8.parquet");
    let rt = sample_runtime();

    let table = Table::from_columns(vec![column_from_slice(&[1i32, 2, 3]).unwrap()]).unwrap();
    let write_opts = parquet::ParquetTableWriteOptions::builder().filepath(&path).column_names(&["x"]).build();
    parquet::write_parquet(&table, &write_opts).unwrap();

    arm_next_io_metadata_fault(cudf_sys::CUDF_TEST_IO_METADATA_FAULT_INVALID_UTF8_NAME);
    let err = match parquet::read_parquet_table(
        &parquet::ParquetTableReadOptions::builder().filepath(&path).with_runtime(&rt).build(),
    ) {
        Ok(_) => panic!("malformed metadata should fail"),
        Err(err) => err,
    };

    assert_eq!(err.kind(), ErrorKind::InvariantViolation);
    assert_eq!(err.code(), ErrorCode::ForeignInvariantViolation);
    assert_eq!(err.operation(), "io::parquet::read_parquet_table");
    assert_eq!(err.fact_value(FactKeyName::ColumnIndex), Some(&FactValue::Usize(0)));
    assert_eq!(error_context_value(&err, "column_name_count").as_deref(), Some("1"));
    assert_eq!(error_context_value(&err, "column_index").as_deref(), Some("0"));
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("column_name"));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("invalid_utf8"));
    assert_eq!(
        error_context_value(&err, "remedy").as_deref(),
        Some("ensure the foreign reader returns valid UTF-8 column names")
    );
    assert_eq!(error_context_value(&err, "name_index").as_deref(), None);
}

#[test]
fn test_parquet_read_missing_projected_column_uses_canonical_column_name_count_context() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("missing_projected_column.parquet");
    let rt = sample_runtime();
    std::fs::write(&path, b"not a parquet file").unwrap();

    let err = match parquet::read_parquet_table(
        &parquet::ParquetTableReadOptions::builder().filepath(&path).columns(&["x"]).with_runtime(&rt).build(),
    ) {
        Ok(_) => panic!("invalid parquet input should fail"),
        Err(err) => err,
    };

    assert_eq!(err.operation(), "io::parquet::read_parquet_table");
    assert_eq!(error_context_value(&err, "column_name_count").as_deref(), Some("1"));
    assert_eq!(error_context_value(&err, "selected_column_count").as_deref(), None);
}

#[test]
fn test_parquet_read_options_can_be_reused_across_calls() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test_reuse_opts.parquet");
    let rt = sample_runtime();

    let c1 = column_from_slice(&[1i32, 2, 3]).unwrap();
    let c2 = column_from_slice(&[4i32, 5, 6]).unwrap();
    let table = Table::from_columns(vec![c1, c2]).unwrap();

    let write_opts = parquet::ParquetTableWriteOptions::builder().filepath(&path).column_names(&["a", "b"]).build();
    parquet::write_parquet(&table, &write_opts).unwrap();

    let read_opts =
        parquet::ParquetTableReadOptions::builder().filepath(&path).columns(&["b"]).with_runtime(&rt).build();

    let first = parquet::read_parquet_table(&read_opts).unwrap();
    let second = parquet::read_parquet_table(&read_opts).unwrap();

    assert_eq!(first.column_names, Some(vec!["b".to_string()]));
    assert_eq!(second.column_names, Some(vec!["b".to_string()]));
    assert_eq!(first.table.testing_view().unwrap().column(0).unwrap().to_vec::<i32>().unwrap(), vec![4, 5, 6]);
    assert_eq!(second.table.testing_view().unwrap().column(0).unwrap().to_vec::<i32>().unwrap(), vec![4, 5, 6]);
}

#[test]
fn test_csv_read_nonexistent_file_reports_not_found() {
    let rt = sample_runtime();
    let read_opts = csv::CsvTableReadOptions::builder()
        .filepath("/tmp/nonexistent_cudf_test_file_12345.csv")
        .with_runtime(&rt)
        .build();
    let err = match csv::read_csv(&read_opts) {
        Ok(_) => panic!("missing CSV file should fail"),
        Err(err) => err,
    };

    // Local filesystem IO now classifies as the domestic `IoFailure` leaf;
    // the coarse `DependencyFailure` kind is the policy bucket.
    assert_eq!(err.code(), ErrorCode::IoFailure);
    assert_eq!(err.kind(), ErrorKind::DependencyFailure);
}

#[test]
fn test_parquet_read_nonexistent_file_reports_not_found() {
    let rt = sample_runtime();
    let read_opts = parquet::ParquetTableReadOptions::builder()
        .filepath("/tmp/nonexistent_cudf_test_file_12345.parquet")
        .with_runtime(&rt)
        .build();
    let err = match parquet::read_parquet_table(&read_opts) {
        Ok(_) => panic!("missing Parquet file should fail"),
        Err(err) => err,
    };

    assert_eq!(err.code(), ErrorCode::IoFailure);
    assert_eq!(err.kind(), ErrorKind::DependencyFailure);
}

#[cfg(unix)]
#[test]
fn test_csv_read_rejects_non_utf8_path_using_stable_contract() {
    let rt = sample_runtime();
    let path = std::ffi::OsString::from_vec(vec![0x66, 0x6f, 0x80, 0x6f]);
    let read_opts =
        csv::CsvTableReadOptions::builder().filepath(std::path::PathBuf::from(path)).with_runtime(&rt).build();
    let err = match csv::read_csv(&read_opts) {
        Ok(_) => panic!("non-UTF8 paths should fail before filesystem access"),
        Err(err) => err,
    };

    // Non-UTF-8 path is a local-IO precondition failure; classified as the
    // domestic `IoFailure` leaf (the kind projection is `DependencyFailure`).
    assert_eq!(err.code(), ErrorCode::IoFailure);
    assert_eq!(err.kind(), ErrorKind::DependencyFailure);
    assert_eq!(err.operation(), "io::csv::read_csv");
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("path"));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("invalid_utf8"));
    assert_eq!(error_context_value(&err, "path").as_deref(), None);
}

#[test]
fn test_csv_read_rejects_path_with_interior_nul_byte_using_stable_contract() {
    let rt = sample_runtime();
    let read_opts = csv::CsvTableReadOptions::builder().filepath("bad\0path.csv").with_runtime(&rt).build();
    let err = match csv::read_csv(&read_opts) {
        Ok(_) => panic!("paths containing NUL should fail before filesystem access"),
        Err(err) => err,
    };

    assert_eq!(err.code(), ErrorCode::NullUnexpected);
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.operation(), "io::csv::read_csv");
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("path"));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("interior_null_byte"));
    assert_eq!(error_context_value(&err, "path").as_deref(), Some("bad\0path.csv"));
}

#[test]
fn test_csv_custom_delimiter() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test_tab.csv");
    let rt = sample_runtime();

    let c1 = column_from_slice(&[1i32, 2]).unwrap();
    let table = Table::from_columns(vec![c1]).unwrap();

    let write_opts = csv::CsvTableWriteOptions::builder().filepath(&path).column_names(&["x"]).delimiter(b'\t').build();
    csv::write_csv(&table, &write_opts).unwrap();

    let read_opts = csv::CsvTableReadOptions::builder().filepath(&path).delimiter(b'\t').with_runtime(&rt).build();
    let result = csv::read_csv(&read_opts).unwrap();
    assert_eq!(result.column_names, Some(vec!["x".to_string()]));
    assert_eq!(result.table.testing_view().unwrap().num_rows().unwrap(), 2);
}
