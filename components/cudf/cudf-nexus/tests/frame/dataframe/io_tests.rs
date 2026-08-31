//! Tests for DataFrame I/O integration.
use std::fs;

use cudf::{
    BoundaryKind, DataFrame, ErrorCode, ErrorKind,
    core::{
        Table,
        error::{FactKeyName, FactValue},
    },
    frame::dataframe::ParquetReadOptions,
    runtime::QueryStageKind,
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

#[test]
fn test_read_parquet_returns_dataframe_with_schema() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test_read.parquet");
    let rt = sample_runtime();

    // Write a parquet file with known column names using the low-level API.
    let c1 = column_from_slice(&[1i32, 2, 3]).unwrap();
    let c2 = column_from_slice(&[10.0f64, 20.0, 30.0]).unwrap();
    let table = Table::from_columns(vec![c1, c2]).unwrap();
    let write_opts =
        cudf::io::parquet::ParquetTableWriteOptions::builder().filepath(&path).column_names(&["id", "value"]).build();
    cudf::io::parquet::write_parquet(&table, &write_opts).unwrap();

    cudf::testing::boundary::reset_boundary_observation_for_test();
    // Read back via DataFrame.
    let df = DataFrame::read_parquet(&rt, &path).unwrap();
    let observation = cudf::testing::boundary::last_boundary_observation_for_test().unwrap();

    assert_eq!(df.num_columns(), 2);
    assert_eq!(df.num_rows(), 3);
    assert_eq!(df.column_names(), vec!["id", "value"]);
    assert_eq!(df.testing_execution_domain_id(), rt.testing_execution_domain_id());
    assert_eq!(df.testing_execution_domain_stream_value(), rt.stream().testing_raw_handle_value());
    assert_eq!(observation.kind, BoundaryKind::HostToDevice);
    assert_eq!(observation.query_stage, Some(QueryStageKind::SourceRead));
    assert_eq!(observation.rows, 3);
    assert_eq!(observation.columns, 2);
    assert_eq!(observation.memory.uploaded_bytes, observation.memory.payload_bytes);
    assert!(observation.memory.payload_bytes.unwrap() > 0);

    let col = df.series("id").unwrap();
    assert_eq!(col.to_vec_strict::<i32>().unwrap(), vec![1, 2, 3]);

    let col = df.series("value").unwrap();
    assert_eq!(col.to_vec_strict::<f64>().unwrap(), vec![10.0, 20.0, 30.0]);
}

#[test]
fn test_read_parquet_with_options() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test_opts.parquet");
    let rt = sample_runtime();

    let c1 = column_from_slice(&[1i32, 2, 3, 4, 5]).unwrap();
    let c2 = column_from_slice(&[10i32, 20, 30, 40, 50]).unwrap();
    let table = Table::from_columns(vec![c1, c2]).unwrap();
    let write_opts =
        cudf::io::parquet::ParquetTableWriteOptions::builder().filepath(&path).column_names(&["a", "b"]).build();
    cudf::io::parquet::write_parquet(&table, &write_opts).unwrap();

    // Read with column selection and row limit.
    let opts = ParquetReadOptions::builder().filepath(&path).columns(&["b"]).num_rows(3).with_runtime(&rt).build();
    let df = DataFrame::read_parquet_with_options(opts).unwrap();

    assert_eq!(df.num_columns(), 1);
    assert_eq!(df.num_rows(), 3);
    assert_eq!(df.column_names(), vec!["b"]);

    let col = df.series("b").unwrap();
    assert_eq!(col.to_vec_strict::<i32>().unwrap(), vec![10, 20, 30]);
}

#[test]
fn test_read_parquet_from_host_buffer_uses_borrowed_bytes_without_retaining_them() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test_host_buffer.parquet");
    let rt = sample_runtime();

    let c1 = column_from_slice(&[1i32, 2, 3, 4, 5]).unwrap();
    let c2 = column_from_slice(&[10i32, 20, 30, 40, 50]).unwrap();
    let table = Table::from_columns(vec![c1, c2]).unwrap();
    let write_opts =
        cudf::io::parquet::ParquetTableWriteOptions::builder().filepath(&path).column_names(&["a", "b"]).build();
    cudf::io::parquet::write_parquet(&table, &write_opts).unwrap();

    let mut filebytes = fs::read(&path).unwrap();
    let opts = ParquetReadOptions::builder().columns(&["b"]).num_rows(3).with_runtime(&rt).build();
    let df = DataFrame::read_parquet_from_host_buffer(&filebytes, opts).unwrap();

    filebytes.fill(0);
    drop(filebytes);

    assert_eq!(df.column_names(), vec!["b"]);
    assert_eq!(df.series("b").unwrap().to_vec_strict::<i32>().unwrap(), vec![10, 20, 30]);
}

#[test]
fn test_read_parquet_with_options_accepts_filebytes_source() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test_filebytes_option.parquet");
    let rt = sample_runtime();

    let table = Table::from_columns(vec![column_from_slice(&[7i32, 8, 9]).unwrap()]).unwrap();
    let write_opts =
        cudf::io::parquet::ParquetTableWriteOptions::builder().filepath(&path).column_names(&["id"]).build();
    cudf::io::parquet::write_parquet(&table, &write_opts).unwrap();

    let filebytes = fs::read(&path).unwrap();
    let opts = ParquetReadOptions::builder().filebytes(&filebytes).with_runtime(&rt).build();
    let df = DataFrame::read_parquet_with_options(opts).unwrap();

    assert_eq!(df.column_names(), vec!["id"]);
    assert_eq!(df.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![7, 8, 9]);
}

#[test]
fn test_gds_driver_available_probe_returns_structured_result() {
    if let Err(err) = cudf::io::gds_driver_available() {
        assert_eq!(err.operation(), "io::gds_driver_available");
        assert_eq!(error_context_value(&err, "resource").as_deref(), Some("cufile_driver"));
    }
}

#[test]
fn test_read_parquet_with_options_rejects_empty_projection_as_invalid_input() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test_empty_projection.parquet");
    let rt = sample_runtime();

    let table = Table::from_columns(vec![column_from_slice(&[1i32, 2]).unwrap()]).unwrap();
    let write_opts =
        cudf::io::parquet::ParquetTableWriteOptions::builder().filepath(&path).column_names(&["id"]).build();
    cudf::io::parquet::write_parquet(&table, &write_opts).unwrap();

    let opts = ParquetReadOptions::builder().filepath(&path).columns(&[]).with_runtime(&rt).build();
    let err = DataFrame::read_parquet_with_options(opts).expect_err("empty projection should fail clearly");

    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.code(), ErrorCode::EmptyInput);
    assert_eq!(err.operation(), "dataframe::read_parquet_with_options");
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("columns"));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("empty_projection"));
    assert_eq!(error_context_value(&err, "column_name_count").as_deref(), Some("0"));
    assert_eq!(
        error_context_value(&err, "remedy").as_deref(),
        Some("omit the projection to read all columns, or provide at least one column name")
    );
}

#[test]
fn test_write_parquet_preserves_schema() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test_write.parquet");
    let rt = sample_runtime();

    let c1 = column_from_slice(&[100i32, 200]).unwrap();
    let c2 = column_from_slice(&[1.5f64, 2.5]).unwrap();
    let df = crate::common::dataframe_from_columns(vec!["x".to_string(), "y".to_string()], vec![c1, c2]).unwrap();

    df.write_parquet(&path).unwrap();

    // Read back and verify schema preserved.
    let df2 = DataFrame::read_parquet(&rt, &path).unwrap();
    assert_eq!(df2.column_names(), vec!["x", "y"]);
    assert_eq!(df2.num_rows(), 2);

    let col = df2.series("x").unwrap();
    assert_eq!(col.to_vec_strict::<i32>().unwrap(), vec![100, 200]);

    let col = df2.series("y").unwrap();
    assert_eq!(col.to_vec_strict::<f64>().unwrap(), vec![1.5, 2.5]);
}

#[test]
fn test_roundtrip_create_write_read() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("roundtrip.parquet");
    let rt = sample_runtime();

    // Create DataFrame -> write -> read -> verify identical.
    let c1 = column_from_slice(&[10i64, 20, 30]).unwrap();
    let c2 = column_from_slice(&[1.0f32, 2.0, 3.0]).unwrap();
    let c3 = column_from_slice(&[7i32, 8, 9]).unwrap();
    let df = crate::common::dataframe_from_columns(
        vec!["alpha".to_string(), "beta".to_string(), "gamma".to_string()],
        vec![c1, c2, c3],
    )
    .unwrap();

    df.write_parquet(&path).unwrap();

    let df2 = DataFrame::read_parquet(&rt, &path).unwrap();
    assert_eq!(df2.column_names(), vec!["alpha", "beta", "gamma"]);
    assert_eq!(df2.num_columns(), 3);
    assert_eq!(df2.num_rows(), 3);

    assert_eq!(df2.series("alpha").unwrap().to_vec_strict::<i64>().unwrap(), vec![10, 20, 30]);
    assert_eq!(df2.series("beta").unwrap().to_vec_strict::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
    assert_eq!(df2.series("gamma").unwrap().to_vec_strict::<i32>().unwrap(), vec![7, 8, 9]);
}

#[test]
fn test_dataframe_read_parquet_nonexistent_file_reports_not_found() {
    let rt = sample_runtime();
    let err = DataFrame::read_parquet(&rt, "/tmp/nonexistent_cudf_test_file_12345.parquet")
        .expect_err("missing Parquet file should fail");

    assert_eq!(err.code(), ErrorCode::IoFailure);
    assert_eq!(err.kind(), ErrorKind::DependencyFailure);
}

#[test]
fn test_dataframe_read_parquet_rejects_missing_schema_metadata_for_nonempty_table() {
    let _reset = MetadataFaultReset;
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("missing_names.parquet");
    let rt = sample_runtime();

    let table = Table::from_columns(vec![column_from_slice(&[1i32, 2, 3]).unwrap()]).unwrap();
    let write_opts =
        cudf::io::parquet::ParquetTableWriteOptions::builder().filepath(&path).column_names(&["x"]).build();
    cudf::io::parquet::write_parquet(&table, &write_opts).unwrap();

    arm_next_io_metadata_fault(cudf_sys::test_support::CUDF_TEST_IO_METADATA_FAULT_ZERO_NAME_COUNT);
    let err = DataFrame::read_parquet(&rt, &path).expect_err("missing schema metadata should fail");

    assert_eq!(err.kind(), ErrorKind::InvariantViolation);
    assert_eq!(err.code(), ErrorCode::ForeignInvariantViolation);
    assert_eq!(err.operation(), "dataframe::read_parquet_with_options");
    assert_eq!(err.fact_value(FactKeyName::ActualCount), Some(&FactValue::Usize(0)));
    assert_eq!(err.fact_value(FactKeyName::TableColumnCount), Some(&FactValue::Usize(1)));
    assert_eq!(error_context_value(&err, "column_name_count").as_deref(), Some("0"));
    assert_eq!(error_context_value(&err, "table_column_count").as_deref(), Some("1"));
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("column_name_count"));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("missing_reader_schema_names"));
    assert_eq!(
        error_context_value(&err, "remedy").as_deref(),
        Some("ensure the reader returns one schema name per table column for non-empty results")
    );
}

#[test]
fn test_dataframe_read_parquet_metadata_invalid_utf8_name_uses_canonical_fault_context() {
    let _reset = MetadataFaultReset;
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("invalid_utf8_name.parquet");
    let rt = sample_runtime();

    let table = Table::from_columns(vec![column_from_slice(&[1i32, 2, 3]).unwrap()]).unwrap();
    let write_opts =
        cudf::io::parquet::ParquetTableWriteOptions::builder().filepath(&path).column_names(&["x"]).build();
    cudf::io::parquet::write_parquet(&table, &write_opts).unwrap();

    arm_next_io_metadata_fault(cudf_sys::test_support::CUDF_TEST_IO_METADATA_FAULT_INVALID_UTF8_NAME);
    let err = DataFrame::read_parquet(&rt, &path).expect_err("invalid UTF-8 metadata should fail");

    assert_eq!(err.kind(), ErrorKind::InvariantViolation);
    assert_eq!(err.code(), ErrorCode::ForeignInvariantViolation);
    assert_eq!(err.operation(), "dataframe::read_parquet_with_options");
    assert!(
        err.call_chain().contains(&"io::parquet::read_parquet_table"),
        "expected reanchor trail to include io::parquet::read_parquet_table, got {:?}",
        err.call_chain()
    );
    assert_eq!(err.fact_value(FactKeyName::ColumnIndex), Some(&FactValue::Usize(0)));
    assert_eq!(error_context_value(&err, "column_name_count").as_deref(), Some("1"));
    assert_eq!(error_context_value(&err, "column_index").as_deref(), Some("0"));
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("column_name"));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("invalid_utf8"));
    assert_eq!(error_context_value(&err, "name_index").as_deref(), None);
}
