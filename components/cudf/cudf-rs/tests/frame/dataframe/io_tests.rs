//! Tests for DataFrame I/O integration.
#[cfg(unix)]
use std::os::unix::ffi::OsStringExt;

use cudf::{
    BoundaryKind, DataFrame, ErrorCode, ErrorKind,
    core::{
        Table,
        error::{FactKeyName, FactValue},
    },
    frame::dataframe::{CsvReadOptions, CsvWriteOptions, ParquetReadOptions},
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
fn test_read_csv_returns_dataframe_with_schema() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test_read.csv");
    let rt = sample_runtime();

    std::fs::write(
        &path,
        "id,value
1,10
2,20
3,30
",
    )
    .unwrap();

    let df = DataFrame::read_csv(&rt, &path).unwrap();

    assert_eq!(df.num_columns(), 2);
    assert_eq!(df.num_rows(), 3);
    assert_eq!(df.column_names(), vec!["id", "value"]);
    assert_eq!(df.testing_execution_domain_id(), rt.testing_execution_domain_id());
    assert_eq!(df.testing_execution_domain_stream_value(), rt.stream().testing_raw_handle_value());
    // CSV read produces nullable columns; use to_optional_vec.
    assert_eq!(df.series("id").unwrap().to_optional_vec::<i64>().unwrap(), vec![Some(1), Some(2), Some(3)]);
    assert_eq!(df.series("value").unwrap().to_optional_vec::<i64>().unwrap(), vec![Some(10), Some(20), Some(30)]);
}

#[test]
fn test_read_csv_with_options_supports_delimiter_and_explicit_names() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test_opts.csv");
    let rt = sample_runtime();

    std::fs::write(
        &path,
        "1\t10
2\t20
3\t30
",
    )
    .unwrap();

    let opts = CsvReadOptions::builder()
        .filepath(&path)
        .delimiter(b'\t')
        .has_header(false)
        .column_names(&["left", "right"])
        .with_runtime(&rt)
        .build();
    let df = DataFrame::read_csv_with_options(opts).unwrap();

    assert_eq!(df.num_columns(), 2);
    assert_eq!(df.num_rows(), 3);
    assert_eq!(df.column_names(), vec!["left", "right"]);
    // CSV read produces nullable columns; use to_optional_vec.
    assert_eq!(df.series("left").unwrap().to_optional_vec::<i64>().unwrap(), vec![Some(1), Some(2), Some(3)]);
    assert_eq!(df.series("right").unwrap().to_optional_vec::<i64>().unwrap(), vec![Some(10), Some(20), Some(30)]);
}

#[test]
fn test_read_csv_with_options_rejects_explicit_name_count_mismatch_with_stable_contract() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("bad_names.csv");
    let rt = sample_runtime();

    std::fs::write(
        &path,
        "1,10
2,20
3,30
",
    )
    .unwrap();

    let opts =
        CsvReadOptions::builder().filepath(&path).has_header(false).column_names(&["only"]).with_runtime(&rt).build();
    let err = DataFrame::read_csv_with_options(opts).expect_err("explicit column name count mismatch should fail");

    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.code(), ErrorCode::ColumnLengthMismatch);
    assert_eq!(err.operation(), "dataframe::read_csv_with_options");
    assert_eq!(error_context_value(&err, "column_name_count").as_deref(), Some("1"));
    assert_eq!(error_context_value(&err, "table_column_count").as_deref(), Some("2"));
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("column_name_count"));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("mismatch_with_table_column_count"));
    assert_eq!(
        error_context_value(&err, "remedy").as_deref(),
        Some("ensure column_name_count matches table_column_count")
    );
}

#[test]
fn test_read_csv_without_header_uses_generated_names() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test_no_header.csv");
    let rt = sample_runtime();

    std::fs::write(
        &path,
        "4,40
5,50
",
    )
    .unwrap();

    let opts = CsvReadOptions::builder().filepath(&path).has_header(false).with_runtime(&rt).build();
    let df = DataFrame::read_csv_with_options(opts).unwrap();

    assert_eq!(df.column_names(), vec!["column_0", "column_1"]);
    // CSV read produces nullable columns; use to_optional_vec.
    assert_eq!(df.series("column_0").unwrap().to_optional_vec::<i64>().unwrap(), vec![Some(4), Some(5)]);
    assert_eq!(df.series("column_1").unwrap().to_optional_vec::<i64>().unwrap(), vec![Some(40), Some(50)]);
}

#[test]
fn test_read_csv_without_header_generates_fallback_names_only_at_dataframe_layer() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test_no_header_contract.csv");
    let rt = sample_runtime();

    std::fs::write(
        &path,
        "11,110
12,120
",
    )
    .unwrap();

    let table_result = cudf::io::csv::read_csv(
        &cudf::io::csv::CsvTableReadOptions::builder().filepath(&path).has_header(false).with_runtime(&rt).build(),
    )
    .unwrap();
    assert_eq!(table_result.column_names, None);

    let df = DataFrame::read_csv_with_options(
        CsvReadOptions::builder().filepath(&path).has_header(false).with_runtime(&rt).build(),
    )
    .unwrap();

    assert_eq!(df.column_names(), vec!["column_0", "column_1"]);
    // CSV read produces nullable columns; use to_optional_vec.
    assert_eq!(df.series("column_0").unwrap().to_optional_vec::<i64>().unwrap(), vec![Some(11), Some(12)]);
    assert_eq!(df.series("column_1").unwrap().to_optional_vec::<i64>().unwrap(), vec![Some(110), Some(120)]);
}

#[test]
fn test_read_csv_explicit_names_override_header_names() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test_override.csv");
    let rt = sample_runtime();

    std::fs::write(
        &path,
        "old_a,old_b
7,70
8,80
",
    )
    .unwrap();

    let opts = CsvReadOptions::builder().filepath(&path).column_names(&["new_a", "new_b"]).with_runtime(&rt).build();
    let df = DataFrame::read_csv_with_options(opts).unwrap();

    assert_eq!(df.column_names(), vec!["new_a", "new_b"]);
    // CSV read-back produces nullable columns; use to_optional_vec.
    assert_eq!(df.series("new_a").unwrap().to_optional_vec::<i64>().unwrap(), vec![Some(7), Some(8)]);
    assert_eq!(df.series("new_b").unwrap().to_optional_vec::<i64>().unwrap(), vec![Some(70), Some(80)]);
}

#[test]
fn test_write_csv_preserves_dataframe_schema() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test_write.csv");
    let rt = sample_runtime();

    let ids = crate::common::series::<i32>("source_ids", &[100, 200]).unwrap();
    let values = crate::common::series::<i32>("source_values", &[1, 2]).unwrap();
    let df = DataFrame::from_columns([("x", ids), ("y", values)]).unwrap();

    df.write_csv(&path).unwrap();

    let written = std::fs::read_to_string(&path).unwrap();
    assert!(written.starts_with(
        "x,y
"
    ));

    let df2 = DataFrame::read_csv(&rt, &path).unwrap();
    assert_eq!(df2.column_names(), vec!["x", "y"]);
    // CSV read-back produces nullable columns; use to_optional_vec.
    assert_eq!(df2.series("x").unwrap().to_optional_vec::<i64>().unwrap(), vec![Some(100), Some(200)]);
    assert_eq!(df2.series("y").unwrap().to_optional_vec::<i64>().unwrap(), vec![Some(1), Some(2)]);
}

#[test]
fn test_write_csv_with_options_supports_delimiter_and_header_toggle() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test_write_opts.csv");
    let rt = sample_runtime();

    let left = crate::common::series::<i32>("left_src", &[9, 8]).unwrap();
    let right = crate::common::series::<i32>("right_src", &[90, 80]).unwrap();
    let df = DataFrame::from_columns([("left", left), ("right", right)]).unwrap();

    let opts = CsvWriteOptions::builder().filepath(&path).delimiter(b'\t').include_header(false).build();
    df.write_csv_with_options(opts).unwrap();

    let written = std::fs::read_to_string(&path).unwrap();
    assert!(written.starts_with(
        "9\t90
"
    ));

    let read_opts = CsvReadOptions::builder()
        .filepath(&path)
        .delimiter(b'\t')
        .has_header(false)
        .column_names(&["left", "right"])
        .with_runtime(&rt)
        .build();
    let df2 = DataFrame::read_csv_with_options(read_opts).unwrap();

    assert_eq!(df2.column_names(), vec!["left", "right"]);
    // CSV read-back produces nullable columns; use to_optional_vec.
    assert_eq!(df2.series("left").unwrap().to_optional_vec::<i64>().unwrap(), vec![Some(9), Some(8)]);
    assert_eq!(df2.series("right").unwrap().to_optional_vec::<i64>().unwrap(), vec![Some(90), Some(80)]);
}

#[test]
fn test_dataframe_read_csv_nonexistent_file_reports_not_found() {
    let rt = sample_runtime();
    let err = DataFrame::read_csv(&rt, "/tmp/nonexistent_cudf_test_file_12345.csv")
        .expect_err("missing CSV file should fail");

    assert_eq!(err.code(), ErrorCode::IoFailure);
    assert_eq!(err.kind(), ErrorKind::DependencyFailure);
}

#[test]
fn test_dataframe_read_parquet_nonexistent_file_reports_not_found() {
    let rt = sample_runtime();
    let err = DataFrame::read_parquet(&rt, "/tmp/nonexistent_cudf_test_file_12345.parquet")
        .expect_err("missing Parquet file should fail");

    assert_eq!(err.code(), ErrorCode::IoFailure);
    assert_eq!(err.kind(), ErrorKind::DependencyFailure);
}

#[cfg(unix)]
#[test]
fn test_dataframe_read_csv_rejects_non_utf8_path_with_high_level_operation_context() {
    let rt = sample_runtime();
    let path = std::ffi::OsString::from_vec(vec![0x66, 0x6f, 0x80, 0x6f]);
    let err = DataFrame::read_csv(&rt, std::path::PathBuf::from(path))
        .expect_err("non-UTF8 paths should fail before filesystem access");

    // Non-UTF-8 path is now classified as the domestic `IoFailure` leaf.
    assert_eq!(err.code(), ErrorCode::IoFailure);
    assert_eq!(err.kind(), ErrorKind::DependencyFailure);
    assert_eq!(err.operation(), "dataframe::read_csv_with_options");
    assert!(
        err.call_chain().contains(&"io::csv::read_csv"),
        "expected reanchor trail to include io::csv::read_csv, got {:?}",
        err.call_chain()
    );
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("path"));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("invalid_utf8"));
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

    arm_next_io_metadata_fault(cudf_sys::CUDF_TEST_IO_METADATA_FAULT_ZERO_NAME_COUNT);
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

    arm_next_io_metadata_fault(cudf_sys::CUDF_TEST_IO_METADATA_FAULT_INVALID_UTF8_NAME);
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
