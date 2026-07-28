//! Tests for JSON / JSON Lines (NDJSON) I/O.
//!
//! All tests require a GPU.
use cudf::{
    ErrorCode, ErrorKind,
    core::{
        Table,
        error::{FactKeyName, FactValue},
    },
    io::json::{self, JsonTableReadOptions, JsonTableWriteOptions},
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

// ===========================================================================
// JSON Lines (NDJSON) round-trip
// ===========================================================================

#[test]
fn test_json_lines_roundtrip_i32() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.jsonl");
    let rt = sample_runtime();

    // Write
    let c1 = column_from_slice(&[1i32, 2, 3]).unwrap();
    let c2 = column_from_slice(&[10.0f64, 20.0, 30.0]).unwrap();
    let table = Table::from_columns(vec![c1, c2]).unwrap();

    let write_opts = JsonTableWriteOptions::builder().filepath(&path).column_names(&["ids", "values"]).build();
    json::write_json(&table, &write_opts).unwrap();

    // Read back through the shared low-level result envelope.
    let read_opts = JsonTableReadOptions::builder().filepath(&path).with_runtime(&rt).build();
    let result = json::read_json_table(&read_opts).unwrap();
    let read_view = result.table.testing_view().unwrap();

    assert_eq!(read_view.num_columns().unwrap(), 2);
    assert_eq!(read_view.num_rows().unwrap(), 3);
    assert_eq!(result.column_names, Some(vec!["ids".to_string(), "values".to_string()]));
    assert_eq!(result.table.testing_execution_domain_id(), rt.testing_execution_domain_id());
    assert_eq!(result.table.testing_execution_domain_stream_value(), rt.stream().testing_raw_handle_value());

    let col0 = read_view.column(0).unwrap();
    assert_eq!(col0.to_vec::<i64>().unwrap(), vec![1, 2, 3]); // JSON integers default to i64
}

#[test]
fn test_json_lines_roundtrip_strings() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.jsonl");
    let rt = sample_runtime();

    // Write a NDJSON file manually
    std::fs::write(
        &path,
        "{\"name\":\"alice\",\"age\":30}
{\"name\":\"bob\",\"age\":25}
",
    )
    .unwrap();

    let read_opts = JsonTableReadOptions::builder().filepath(&path).with_runtime(&rt).build();
    let result = json::read_json_table(&read_opts).unwrap();
    let read_view = result.table.testing_view().unwrap();

    assert_eq!(read_view.num_rows().unwrap(), 2);
    let names = result.column_names.expect("json object keys should be exposed as schema metadata");
    assert!(names.contains(&"name".to_string()));
    assert!(names.contains(&"age".to_string()));
}

// ===========================================================================
// Empty file
// ===========================================================================

#[test]
fn test_json_lines_empty_file() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("empty.jsonl");
    let rt = sample_runtime();

    std::fs::write(&path, "").unwrap();

    let read_opts = JsonTableReadOptions::builder().filepath(&path).with_runtime(&rt).build();
    let result = json::read_json_table(&read_opts).unwrap();

    assert_eq!(result.table.testing_view().unwrap().num_rows().unwrap(), 0);
}

// ===========================================================================
// Write options
// ===========================================================================

#[test]
fn test_json_lines_write_defaults_to_ndjson() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.jsonl");

    let c1 = column_from_slice(&[1i32]).unwrap();
    let table = Table::from_columns(vec![c1]).unwrap();

    let write_opts = JsonTableWriteOptions::builder().filepath(&path).column_names(&["x"]).build();
    json::write_json(&table, &write_opts).unwrap();

    // NDJSON: each line should be a JSON object
    let content = std::fs::read_to_string(&path).unwrap();
    assert!(content.contains("\"x\""));
}

#[test]
fn test_json_write_rejects_column_name_with_nul_byte() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("bad.jsonl");

    let c1 = column_from_slice(&[1i32]).unwrap();
    let table = Table::from_columns(vec![c1]).unwrap();

    let write_opts = JsonTableWriteOptions::builder().filepath(&path).column_names(&["bad\0name"]).build();
    let err = json::write_json(&table, &write_opts).unwrap_err();

    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.code(), ErrorCode::NullUnexpected);
    assert_eq!(err.operation(), "io::json::write_json");
    assert_eq!(err.fact_value(FactKeyName::ColumnIndex), Some(&FactValue::Usize(0)));
    assert_eq!(error_context_value(&err, "column_index").as_deref(), Some("0"));
    assert_eq!(error_context_value(&err, "column_name").as_deref(), Some("bad\0name"));
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("column_name"));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("interior_null_byte"));
    assert!(err.to_string().contains("null byte"), "unexpected error: {err}");
}

#[test]
fn test_json_write_rejects_column_name_count_with_stable_contract() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("bad_count.jsonl");

    let c1 = column_from_slice(&[1i32]).unwrap();
    let table = Table::from_columns(vec![c1]).unwrap();

    let write_opts = JsonTableWriteOptions::builder().filepath(&path).column_names(&["only", "extra"]).build();
    let err = json::write_json(&table, &write_opts).expect_err("column name count mismatch should fail before FFI");

    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.code(), ErrorCode::ColumnLengthMismatch);
    assert_eq!(err.operation(), "io::json::write_json");
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

// ===========================================================================
// Nonexistent file error
// ===========================================================================

#[test]
fn test_json_read_nonexistent_file() {
    let rt = sample_runtime();
    let read_opts = JsonTableReadOptions::builder().filepath("/nonexistent/file.jsonl").with_runtime(&rt).build();
    let err = match json::read_json_table(&read_opts) {
        Ok(_) => panic!("missing JSON file should fail"),
        Err(err) => err,
    };

    assert_eq!(err.code(), ErrorCode::IoFailure);
    assert_eq!(err.kind(), ErrorKind::DependencyFailure);
}

#[test]
fn test_json_read_metadata_negative_name_count_maps_to_invariant_violation() {
    let _reset = MetadataFaultReset;
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("metadata_fault.jsonl");
    let rt = sample_runtime();

    std::fs::write(&path, "{\"id\":1,\"value\":10}\n{\"id\":2,\"value\":20}\n").unwrap();

    arm_next_io_metadata_fault(cudf_sys::test_support::CUDF_TEST_IO_METADATA_FAULT_NEGATIVE_NAME_COUNT);
    let err = match json::read_json_table(&JsonTableReadOptions::builder().filepath(&path).with_runtime(&rt).build()) {
        Ok(_) => panic!("malformed metadata should fail"),
        Err(err) => err,
    };

    assert_eq!(err.kind(), ErrorKind::InvariantViolation);
    assert_eq!(err.code(), ErrorCode::ForeignInvariantViolation);
    assert_eq!(err.operation(), "io::json::read_json_table");
    assert_eq!(error_context_value(&err, "column_name_count").as_deref(), Some("-1"));
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("column_name_count"));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("negative_name_count"));
    assert_eq!(
        error_context_value(&err, "remedy").as_deref(),
        Some("ensure the foreign reader reports a non-negative column name count")
    );
    assert_eq!(error_context_value(&err, "num_names").as_deref(), None);
}
