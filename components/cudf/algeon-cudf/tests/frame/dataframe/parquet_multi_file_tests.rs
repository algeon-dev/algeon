//! Tests for high-level multi-file Parquet reads.

use cudf::{
    DataFrame, ErrorCode, ErrorKind,
    core::{Column, Table},
    frame::dataframe::MultiParquetReadOptions,
    io::parquet::{ParquetTableWriteOptions, write_parquet},
    testing::prelude::*,
};
use cudf_sys::test_support;
use tempfile::TempDir;

use crate::common::{
    arm_next_io_metadata_fault, column_from_slice, error_context_value, last_exec_mr_value, last_exec_stream_value,
    reset_exec_observation, reset_io_metadata_fault, sample_runtime,
};

struct MetadataFaultReset;

impl Drop for MetadataFaultReset {
    fn drop(&mut self) {
        reset_io_metadata_fault();
    }
}

fn write_parquet_df(path: &std::path::Path, column_names: &[&str], columns: Vec<Column>) {
    let table = Table::from_columns(columns).unwrap();
    let write_opts = ParquetTableWriteOptions::builder().filepath(path).column_names(column_names).build();
    write_parquet(&table, &write_opts).unwrap();
}

#[test]
fn test_read_parquet_many_preserves_file_order_and_schema() {
    let dir = TempDir::new().unwrap();
    let first_path = dir.path().join("first.parquet");
    let second_path = dir.path().join("second.parquet");
    let rt = sample_runtime();

    write_parquet_df(
        &first_path,
        &["id", "value"],
        vec![column_from_slice(&[1i32, 2]).unwrap(), column_from_slice(&[10i64, 20]).unwrap()],
    );
    write_parquet_df(
        &second_path,
        &["id", "value"],
        vec![column_from_slice(&[3i32, 4]).unwrap(), column_from_slice(&[30i64, 40]).unwrap()],
    );

    let df = DataFrame::read_parquet_many(&rt, &[first_path, second_path]).unwrap();

    assert_eq!(df.column_names(), vec!["id", "value"]);
    assert_eq!(df.num_rows(), 4);
    assert_eq!(df.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![1, 2, 3, 4]);
    assert_eq!(df.series("value").unwrap().to_vec_strict::<i64>().unwrap(), vec![10, 20, 30, 40]);
}

#[test]
fn test_read_parquet_many_projection_applies_to_all_files() {
    let dir = TempDir::new().unwrap();
    let first_path = dir.path().join("project_first.parquet");
    let second_path = dir.path().join("project_second.parquet");
    let rt = sample_runtime();

    write_parquet_df(
        &first_path,
        &["id", "value", "payload"],
        vec![
            column_from_slice(&[1i32, 2]).unwrap(),
            column_from_slice(&[10i64, 20]).unwrap(),
            column_from_slice(&[100i32, 200]).unwrap(),
        ],
    );
    write_parquet_df(
        &second_path,
        &["id", "value", "payload"],
        vec![
            column_from_slice(&[3i32]).unwrap(),
            column_from_slice(&[30i64]).unwrap(),
            column_from_slice(&[300i32]).unwrap(),
        ],
    );

    let opts = MultiParquetReadOptions::builder()
        .filepaths(&[first_path, second_path])
        .columns(&["value"])
        .with_runtime(&rt)
        .build();
    let df = DataFrame::read_parquet_many_with_options(opts).unwrap();

    assert_eq!(df.column_names(), vec!["value"]);
    assert_eq!(df.num_columns(), 1);
    assert_eq!(df.num_rows(), 3);
    assert_eq!(df.series("value").unwrap().to_vec_strict::<i64>().unwrap(), vec![10, 20, 30]);
}

#[test]
fn test_read_parquet_many_rejects_mismatched_schema_with_stable_error() {
    let dir = TempDir::new().unwrap();
    let first_path = dir.path().join("schema_first.parquet");
    let second_path = dir.path().join("schema_second.parquet");
    let rt = sample_runtime();

    write_parquet_df(
        &first_path,
        &["id", "value"],
        vec![column_from_slice(&[1i32]).unwrap(), column_from_slice(&[10i64]).unwrap()],
    );
    write_parquet_df(
        &second_path,
        &["id", "amount"],
        vec![column_from_slice(&[2i32]).unwrap(), column_from_slice(&[20i64]).unwrap()],
    );

    let opts = MultiParquetReadOptions::builder().filepaths(&[&first_path, &second_path]).with_runtime(&rt).build();
    let err = DataFrame::read_parquet_many_with_options(opts).expect_err("schema mismatch should fail");

    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(err.operation(), "dataframe::read_parquet_many_with_options");
    assert_eq!(error_context_value(&err, "left_column").as_deref(), Some("value"));
    assert_eq!(error_context_value(&err, "right_column").as_deref(), Some("amount"));
    assert_eq!(error_context_value(&err, "reference_file_index").as_deref(), Some("0"));
    assert_eq!(error_context_value(&err, "file_index").as_deref(), Some("1"));
    assert_eq!(error_context_value(&err, "reference_path").as_deref(), Some(first_path.to_string_lossy().as_ref()));
    assert_eq!(error_context_value(&err, "path").as_deref(), Some(second_path.to_string_lossy().as_ref()));
    assert!(
        err.call_chain().contains(&"dataframe::concat_many"),
        "expected concat_many in call chain, got {:?}",
        err.call_chain()
    );
}

#[test]
fn test_read_parquet_many_rejects_empty_file_list_as_invalid_input() {
    let rt = sample_runtime();
    let paths: Vec<std::path::PathBuf> = Vec::new();
    let opts = MultiParquetReadOptions::builder().filepaths(&paths).with_runtime(&rt).build();
    let err = DataFrame::read_parquet_many_with_options(opts).expect_err("empty file list should fail");

    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.code(), ErrorCode::EmptyInput);
    assert_eq!(err.operation(), "dataframe::read_parquet_many_with_options");
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("filepaths"));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("empty_input"));
    assert_eq!(error_context_value(&err, "remedy").as_deref(), Some("provide one or more concrete Parquet file paths"));
}

#[test]
fn test_read_parquet_many_rejects_empty_projection_as_invalid_input() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("empty_projection.parquet");
    let rt = sample_runtime();

    write_parquet_df(&path, &["id"], vec![column_from_slice(&[1i32, 2]).unwrap()]);

    let opts = MultiParquetReadOptions::builder().filepaths(&[path]).columns(&[]).with_runtime(&rt).build();
    let err = DataFrame::read_parquet_many_with_options(opts).expect_err("empty projection should fail clearly");

    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.code(), ErrorCode::EmptyInput);
    assert_eq!(err.operation(), "dataframe::read_parquet_many_with_options");
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("columns"));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("empty_projection"));
    assert_eq!(error_context_value(&err, "column_name_count").as_deref(), Some("0"));
    assert_eq!(
        error_context_value(&err, "remedy").as_deref(),
        Some("omit the projection to read all columns, or provide at least one column name")
    );
}

#[test]
fn test_read_parquet_many_rejects_directory_via_shared_input_validator() {
    let dir = TempDir::new().unwrap();
    let rt = sample_runtime();

    let dir_err = DataFrame::read_parquet_many(&rt, [dir.path()]).expect_err("directory input should fail");
    assert_eq!(dir_err.kind(), ErrorKind::InvalidInput);
    assert_eq!(dir_err.code(), ErrorCode::PathIsDirectory);
    assert_eq!(dir_err.operation(), "dataframe::read_parquet_many_with_options");
    assert!(
        dir_err.call_chain().contains(&"io::parquet::read_parquet_table"),
        "expected low-level parquet reader in call chain, got {:?}",
        dir_err.call_chain()
    );
}

#[test]
fn test_read_parquet_many_missing_file_reports_file_index_and_path() {
    let dir = TempDir::new().unwrap();
    let first_path = dir.path().join("existing.parquet");
    let missing_path = dir.path().join("missing.parquet");
    let rt = sample_runtime();

    write_parquet_df(&first_path, &["id"], vec![column_from_slice(&[1i32, 2]).unwrap()]);

    let err = DataFrame::read_parquet_many(&rt, [&first_path, &missing_path])
        .expect_err("missing file should fail with multi-file context");

    assert_eq!(err.kind(), ErrorKind::DependencyFailure);
    assert_eq!(err.code(), ErrorCode::IoFailure);
    assert_eq!(err.operation(), "dataframe::read_parquet_many_with_options");
    assert_eq!(error_context_value(&err, "file_index").as_deref(), Some("1"));
    assert_eq!(error_context_value(&err, "path").as_deref(), Some(missing_path.to_string_lossy().as_ref()));
    assert!(
        err.call_chain().contains(&"io::parquet::read_parquet_table"),
        "expected low-level parquet reader in call chain, got {:?}",
        err.call_chain()
    );
}

#[test]
fn test_read_parquet_many_metadata_fault_reports_file_index_and_path() {
    let _reset = MetadataFaultReset;
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("metadata_fault.parquet");
    let rt = sample_runtime();

    write_parquet_df(&path, &["id"], vec![column_from_slice(&[1i32, 2]).unwrap()]);

    arm_next_io_metadata_fault(test_support::CUDF_TEST_IO_METADATA_FAULT_ZERO_NAME_COUNT);
    let err = DataFrame::read_parquet_many(&rt, [&path]).expect_err("metadata fault should fail");

    assert_eq!(err.kind(), ErrorKind::InvariantViolation);
    assert_eq!(err.code(), ErrorCode::ForeignInvariantViolation);
    assert_eq!(err.operation(), "dataframe::read_parquet_many_with_options");
    assert_eq!(error_context_value(&err, "file_index").as_deref(), Some("0"));
    assert_eq!(error_context_value(&err, "path").as_deref(), Some(path.to_string_lossy().as_ref()));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("missing_reader_schema_names"));
}

#[test]
fn test_read_parquet_many_accepts_literal_glob_metacharacters_in_filename() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("part-[literal]?*.parquet");
    let rt = sample_runtime();

    write_parquet_df(&path, &["id"], vec![column_from_slice(&[7i32, 8]).unwrap()]);

    let df = DataFrame::read_parquet_many(&rt, [&path]).unwrap();

    assert_eq!(df.column_names(), vec!["id"]);
    assert_eq!(df.num_rows(), 2);
    assert_eq!(df.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![7, 8]);
}

#[test]
fn test_read_parquet_many_does_not_expand_glob_patterns() {
    let dir = TempDir::new().unwrap();
    let existing_path = dir.path().join("part-0.parquet");
    let glob_pattern = dir.path().join("*.parquet");
    let rt = sample_runtime();

    write_parquet_df(&existing_path, &["id"], vec![column_from_slice(&[1i32, 2]).unwrap()]);

    let err = DataFrame::read_parquet_many(&rt, [&glob_pattern]).expect_err("glob patterns should not be expanded");

    assert_eq!(err.kind(), ErrorKind::DependencyFailure);
    assert_eq!(err.code(), ErrorCode::IoFailure);
    assert_eq!(err.operation(), "dataframe::read_parquet_many_with_options");
    assert_eq!(error_context_value(&err, "file_index").as_deref(), Some("0"));
    assert_eq!(error_context_value(&err, "path").as_deref(), Some(glob_pattern.to_string_lossy().as_ref()));
}

#[test]
fn test_read_parquet_many_retains_runtime_execution_domain_provenance() {
    let dir = TempDir::new().unwrap();
    let first_path = dir.path().join("domain_first.parquet");
    let second_path = dir.path().join("domain_second.parquet");
    let rt = sample_runtime();
    let expected_stream = rt.stream().testing_raw_handle_value();
    let expected_mr = rt.memory_resource().testing_raw_handle_value();

    write_parquet_df(&first_path, &["id"], vec![column_from_slice(&[1i32, 2]).unwrap()]);
    write_parquet_df(&second_path, &["id"], vec![column_from_slice(&[3i32, 4]).unwrap()]);

    reset_exec_observation();
    let df = DataFrame::read_parquet_many(&rt, &[first_path, second_path]).unwrap();

    assert_eq!(df.testing_execution_domain_id(), rt.testing_execution_domain_id());
    assert_eq!(df.testing_execution_domain_stream_value(), expected_stream);
    assert_eq!(df.testing_execution_domain_mr_value(), expected_mr);
    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_eq!(last_exec_mr_value(), expected_mr);
}
