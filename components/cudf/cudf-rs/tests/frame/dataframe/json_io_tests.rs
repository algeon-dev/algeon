//! Tests for DataFrame-level JSON I/O methods.
//!
//! All tests require a GPU.
use cudf::{
    DataFrame, ErrorKind,
    frame::dataframe::{JsonReadOptions, JsonWriteOptions},
    testing::prelude::*,
};
use tempfile::TempDir;

use crate::common::sample_runtime;

// ===========================================================================
// Convenience methods: read_json / write_json
// ===========================================================================

#[test]
fn test_dataframe_json_roundtrip_convenience() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.jsonl");
    let rt = sample_runtime();

    let s1 = crate::common::series("ids", &[1i32, 2, 3]).unwrap();
    let s2 = crate::common::series("values", &[10.0f64, 20.0, 30.0]).unwrap();
    let df = DataFrame::from_columns([("ids", s1), ("values", s2)]).unwrap();
    df.write_json(&path).unwrap();

    let read_df = DataFrame::read_json(&rt, &path).unwrap();
    assert_eq!(read_df.num_rows(), 3);
    assert_eq!(read_df.num_columns(), 2);
    assert!(read_df.contains("ids"));
    assert!(read_df.contains("values"));
    assert_eq!(read_df.testing_execution_domain_id(), rt.testing_execution_domain_id());
    assert_eq!(read_df.testing_execution_domain_stream_value(), rt.stream().testing_raw_handle_value());
}

#[test]
fn test_dataframe_json_preserves_column_names() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("names.jsonl");
    let rt = sample_runtime();

    let s1 = crate::common::series("alpha", &[100i64]).unwrap();
    let s2 = crate::common::series("beta", &[200i64]).unwrap();
    let s3 = crate::common::series("gamma", &[300i64]).unwrap();
    let df = DataFrame::from_columns([("alpha", s1), ("beta", s2), ("gamma", s3)]).unwrap();
    df.write_json(&path).unwrap();

    let read_df = DataFrame::read_json(&rt, &path).unwrap();
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
    let rt = sample_runtime();

    let s1 = crate::common::series("x", &[42i32]).unwrap();
    let df = DataFrame::from_columns([("x", s1)]).unwrap();

    let write_opts = JsonWriteOptions::builder().filepath(&path).build();
    df.write_json_with_options(write_opts).unwrap();

    let read_opts = JsonReadOptions::builder().filepath(&path).with_runtime(&rt).build();
    let read_df = DataFrame::read_json_with_options(read_opts).unwrap();
    assert_eq!(read_df.num_rows(), 1);
    assert!(read_df.contains("x"));
}

#[test]
fn test_dataframe_json_write_no_nulls_option() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("no_nulls.jsonl");
    let rt = sample_runtime();

    let s1 = crate::common::series("a", &[1i32, 2]).unwrap();
    let df = DataFrame::from_columns([("a", s1)]).unwrap();

    let write_opts = JsonWriteOptions::builder().filepath(&path).include_nulls(false).build();
    df.write_json_with_options(write_opts).unwrap();

    // Read back — should still work
    let read_df = DataFrame::read_json(&rt, &path).unwrap();
    assert_eq!(read_df.num_rows(), 2);
}

// ===========================================================================
// Read from hand-written NDJSON
// ===========================================================================

#[test]
fn test_dataframe_json_read_hand_written_ndjson() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("hand.jsonl");
    let rt = sample_runtime();
    std::fs::write(
        &path,
        "{\"name\":\"alice\",\"age\":30}
{\"name\":\"bob\",\"age\":25}
",
    )
    .unwrap();

    let df = DataFrame::read_json(&rt, &path).unwrap();
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
    let rt = sample_runtime();

    // Write an empty file (no NDJSON rows)
    std::fs::write(&path, "").unwrap();

    let df = DataFrame::read_json(&rt, &path).unwrap();
    assert_eq!(df.num_rows(), 0);
}

// ===========================================================================
// Error: nonexistent file
// ===========================================================================

#[test]
fn test_dataframe_json_read_nonexistent_file() {
    let rt = sample_runtime();
    let err = DataFrame::read_json(&rt, "/nonexistent/file.jsonl").expect_err("missing JSON file should fail");

    assert_eq!(err.kind(), ErrorKind::NotFound);
}
