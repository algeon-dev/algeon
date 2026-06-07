//! Tests for JSON / JSON Lines (NDJSON) I/O.
//!
//! All tests require a GPU.

use cudf::{
    Column, Table,
    io::json::{self, JsonReaderOptions, JsonWriterOptions},
};
use tempfile::TempDir;

// ===========================================================================
// JSON Lines (NDJSON) round-trip
// ===========================================================================

#[test]
fn test_json_lines_roundtrip_i32() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.jsonl");

    // Write
    let c1 = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let c2 = Column::from_slice(&[10.0f64, 20.0, 30.0]).unwrap();
    let table = Table::from_columns(vec![c1, c2]).unwrap();
    let view = table.view().unwrap();

    let write_opts = JsonWriterOptions::builder().filepath(&path).column_names(&["ids", "values"]).build();
    json::write_json(&view, &write_opts).unwrap();

    // Read back with metadata
    let read_opts = JsonReaderOptions::builder().filepath(&path).build();
    let result = json::read_json_with_metadata(&read_opts).unwrap();
    let read_view = result.table.view().unwrap();

    assert_eq!(read_view.num_columns().unwrap(), 2);
    assert_eq!(read_view.num_rows().unwrap(), 3);
    assert_eq!(result.column_names, vec!["ids", "values"]);

    let col0 = read_view.column(0).unwrap();
    assert_eq!(col0.to_vec::<i64>().unwrap(), vec![1, 2, 3]); // JSON integers default to i64
}

#[test]
fn test_json_lines_roundtrip_strings() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.jsonl");

    // Write a NDJSON file manually
    std::fs::write(&path, "{\"name\":\"alice\",\"age\":30}\n{\"name\":\"bob\",\"age\":25}\n").unwrap();

    let read_opts = JsonReaderOptions::builder().filepath(&path).build();
    let result = json::read_json_with_metadata(&read_opts).unwrap();
    let read_view = result.table.view().unwrap();

    assert_eq!(read_view.num_rows().unwrap(), 2);
    assert!(result.column_names.contains(&"name".to_string()));
    assert!(result.column_names.contains(&"age".to_string()));
}

// ===========================================================================
// Read without metadata
// ===========================================================================

#[test]
fn test_json_lines_read_without_metadata() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.jsonl");

    std::fs::write(&path, "{\"x\":1}\n{\"x\":2}\n").unwrap();

    let read_opts = JsonReaderOptions::builder().filepath(&path).build();
    let table = json::read_json(&read_opts).unwrap();
    let view = table.view().unwrap();

    assert_eq!(view.num_columns().unwrap(), 1);
    assert_eq!(view.num_rows().unwrap(), 2);
}

// ===========================================================================
// Empty file
// ===========================================================================

#[test]
fn test_json_lines_empty_file() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("empty.jsonl");

    std::fs::write(&path, "").unwrap();

    let read_opts = JsonReaderOptions::builder().filepath(&path).build();
    let result = json::read_json_with_metadata(&read_opts).unwrap();

    assert_eq!(result.table.view().unwrap().num_rows().unwrap(), 0);
}

// ===========================================================================
// Write options
// ===========================================================================

#[test]
fn test_json_lines_write_defaults_to_ndjson() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.jsonl");

    let c1 = Column::from_slice(&[1i32]).unwrap();
    let table = Table::from_columns(vec![c1]).unwrap();
    let view = table.view().unwrap();

    let write_opts = JsonWriterOptions::builder().filepath(&path).column_names(&["x"]).build();
    json::write_json(&view, &write_opts).unwrap();

    // NDJSON: each line should be a JSON object
    let content = std::fs::read_to_string(&path).unwrap();
    assert!(content.contains("\"x\""));
}

#[test]
fn test_json_write_rejects_column_name_with_nul_byte() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("bad.jsonl");

    let c1 = Column::from_slice(&[1i32]).unwrap();
    let table = Table::from_columns(vec![c1]).unwrap();
    let view = table.view().unwrap();

    let write_opts = JsonWriterOptions::builder().filepath(&path).column_names(&["bad\0name"]).build();
    let err = json::write_json(&view, &write_opts).unwrap_err();

    assert!(err.to_string().contains("null byte"), "unexpected error: {err}");
}

// ===========================================================================
// Nonexistent file error
// ===========================================================================

#[test]
fn test_json_read_nonexistent_file() {
    let read_opts = JsonReaderOptions::builder().filepath("/nonexistent/file.jsonl").build();
    let result = json::read_json(&read_opts);
    assert!(result.is_err());
}
