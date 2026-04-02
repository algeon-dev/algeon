//! Tests for DataFrame-level JSON I/O facade.
//!
//! All tests require a GPU.

use cudf::{DataFrame, JsonReaderOptions, JsonWriteOptions, Series};
use tempfile::TempDir;

// ===========================================================================
// Convenience methods: read_json / write_json
// ===========================================================================

#[test]
fn test_dataframe_json_roundtrip_convenience() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.jsonl");

    let s1 = Series::new("ids", &[1i32, 2, 3]).unwrap();
    let s2 = Series::new("values", &[10.0f64, 20.0, 30.0]).unwrap();
    let df = DataFrame::from_series([("ids", s1), ("values", s2)]).unwrap();
    df.write_json(&path).unwrap();

    let read_df = DataFrame::read_json(&path).unwrap();
    assert_eq!(read_df.num_rows(), 3);
    assert_eq!(read_df.num_columns(), 2);
    assert!(read_df.contains("ids"));
    assert!(read_df.contains("values"));
}

#[test]
fn test_dataframe_json_preserves_column_names() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("names.jsonl");

    let s1 = Series::new("alpha", &[100i64]).unwrap();
    let s2 = Series::new("beta", &[200i64]).unwrap();
    let s3 = Series::new("gamma", &[300i64]).unwrap();
    let df = DataFrame::from_series([("alpha", s1), ("beta", s2), ("gamma", s3)]).unwrap();
    df.write_json(&path).unwrap();

    let read_df = DataFrame::read_json(&path).unwrap();
    let names = read_df.column_names();
    assert!(names.contains(&"alpha"));
    assert!(names.contains(&"beta"));
    assert!(names.contains(&"gamma"));
}

// ===========================================================================
// Options-based methods: read_json_with_options / write_json_with_options
// ===========================================================================

#[test]
fn test_dataframe_json_roundtrip_with_options() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("opts.jsonl");

    let s1 = Series::new("x", &[42i32]).unwrap();
    let df = DataFrame::from_series([("x", s1)]).unwrap();

    let write_opts = JsonWriteOptions::builder().filepath(&path).build();
    df.write_json_with_options(write_opts).unwrap();

    let read_opts = JsonReaderOptions::builder().filepath(&path).build();
    let read_df = DataFrame::read_json_with_options(read_opts).unwrap();
    assert_eq!(read_df.num_rows(), 1);
    assert!(read_df.contains("x"));
}

#[test]
fn test_dataframe_json_write_no_nulls_option() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("no_nulls.jsonl");

    let s1 = Series::new("a", &[1i32, 2]).unwrap();
    let df = DataFrame::from_series([("a", s1)]).unwrap();

    let write_opts = JsonWriteOptions::builder().filepath(&path).include_nulls(false).build();
    df.write_json_with_options(write_opts).unwrap();

    // Read back — should still work
    let read_df = DataFrame::read_json(&path).unwrap();
    assert_eq!(read_df.num_rows(), 2);
}

// ===========================================================================
// Read from hand-written NDJSON
// ===========================================================================

#[test]
fn test_dataframe_json_read_hand_written_ndjson() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("hand.jsonl");
    std::fs::write(&path, "{\"name\":\"alice\",\"age\":30}\n{\"name\":\"bob\",\"age\":25}\n").unwrap();

    let df = DataFrame::read_json(&path).unwrap();
    assert_eq!(df.num_rows(), 2);
    assert!(df.contains("name"));
    assert!(df.contains("age"));
}

// ===========================================================================
// Empty DataFrame
// ===========================================================================

#[test]
fn test_dataframe_json_empty_roundtrip() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("empty.jsonl");

    // Write an empty file (no NDJSON rows)
    std::fs::write(&path, "").unwrap();

    let df = DataFrame::read_json(&path).unwrap();
    assert_eq!(df.num_rows(), 0);
}

// ===========================================================================
// Error: nonexistent file
// ===========================================================================

#[test]
fn test_dataframe_json_read_nonexistent_file() {
    let result = DataFrame::read_json("/nonexistent/file.jsonl");
    assert!(result.is_err());
}
