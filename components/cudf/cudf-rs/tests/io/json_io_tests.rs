//! Tests for JSON / JSON Lines (NDJSON) I/O.
//!
//! All tests require a GPU.
use cudf::{
    ErrorKind,
    core::Table,
    io::json::{self, JsonTableReadOptions, JsonTableWriteOptions},
    testing::prelude::*,
};
use tempfile::TempDir;

use crate::common::{column_from_slice, sample_runtime};

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

    assert!(err.to_string().contains("null byte"), "unexpected error: {err}");
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

    assert_eq!(err.kind(), ErrorKind::NotFound);
}
