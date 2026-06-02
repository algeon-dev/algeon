//! Contract tests for `ParquetChunkedReader`.
//!
//! These tests define the EXPECTED behavior of the chunked Parquet reader
//! BEFORE the implementation exists. All tests are `#[ignore]` until the
//! chunked reader API is implemented.
//!
//! # Expected API (from SI-SHAPE spec)
//!
//! ```rust,ignore
//! // Builder
//! let options = ParquetChunkedReaderOptions::builder()
//!     .path("data.parquet")
//!     .chunk_read_limit(256 * 1024 * 1024)   // bytes of device memory per chunk
//!     .pass_read_limit(512 * 1024 * 1024)    // decompression scratch limit
//!     .columns(&["id", "name", "value"])     // column projection
//!     .build()?;
//!
//! // Reader
//! let mut reader = ParquetChunkedReader::open(options)?;
//! let names: &[String] = reader.column_names();
//!
//! while let Some(df) = reader.next_chunk()? {
//!     process(df);
//! }
//! // Drop releases file handle, decoder state, device memory.
//! ```
//!
//! # Key invariants
//!
//! - `next_chunk() -> Result<Option<DataFrame>>`
//! - `Ok(None)` at EOF (idempotent)
//! - Sticky errors: after first `Err`, all subsequent calls return same error
//! - Reader is `!Send` (must be created and consumed on the same OS thread)
//! - Returned `DataFrame`s are independent owned values (`Send + Sync`)
//! - Early drop is safe; previously returned DataFrames remain valid

use std::path::Path;

use cudf::{
    column::Column,
    dataframe::DataFrame,
    io::parquet::{ParquetWriterOptions, write_parquet},
    table::Table,
};
use tempfile::TempDir;

// Future chunked reader types (uncomment when implemented):
// use cudf::io::parquet::{ParquetChunkedReader, ParquetChunkedReaderOptions};

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

/// Write a test Parquet file with `num_columns` i32 columns and `num_rows` rows.
///
/// Column names are "c0", "c1", ..., "c{n-1}".
/// Column `ci` contains values `[i, i+num_columns, i+2*num_columns, ...]` so
/// each column has distinct, verifiable data.
fn write_test_parquet(path: &Path, num_rows: i32, num_columns: usize) {
    let mut columns = Vec::with_capacity(num_columns);
    let mut names: Vec<&str> = Vec::with_capacity(num_columns);

    // We need to hold the name strings so they live long enough.
    let name_strings: Vec<String> = (0..num_columns).map(|i| format!("c{}", i)).collect();

    for i in 0..num_columns {
        let data: Vec<i32> = (0..num_rows).map(|r| r * num_columns as i32 + i as i32).collect();
        columns.push(Column::from_slice(&data).unwrap());
        names.push(&name_strings[i]);
    }

    let table = Table::from_columns(columns).unwrap();
    let view = table.view().unwrap();
    let opts = ParquetWriterOptions::new(path).column_names(&names).unwrap();
    write_parquet(&view, &opts).unwrap();
}

/// Write a larger test Parquet file suitable for multi-chunk reads.
///
/// Creates 3 columns × `num_rows` rows of i32 data.
fn write_large_test_parquet(path: &Path, num_rows: i32) {
    write_test_parquet(path, num_rows, 3);
}

/// Read the entire file as a single-shot DataFrame for comparison.
fn read_single_shot(path: &Path) -> DataFrame {
    DataFrame::read_parquet(path).unwrap()
}

// ---------------------------------------------------------------------------
// Category 1: Basic chunked read
// ---------------------------------------------------------------------------

/// Concatenate all chunks from a chunked read and verify that the total row
/// count matches a single-shot read of the same file.
#[test]
#[ignore = "chunked reader not yet implemented"]
fn chunked_read_produces_all_rows() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("all_rows.parquet");
    write_large_test_parquet(&path, 10_000);

    let expected = read_single_shot(&path);

    // -- chunked read --
    // let opts = ParquetChunkedReaderOptions::builder()
    //     .path(&path)
    //     .chunk_read_limit(4096) // small limit to force multiple chunks
    //     .build()
    //     .unwrap();
    // let mut reader = ParquetChunkedReader::open(opts).unwrap();
    //
    // let mut total_rows = 0usize;
    // let mut chunk_count = 0usize;
    // while let Some(df) = reader.next_chunk().unwrap() {
    //     assert!(df.num_rows() > 0, "chunk must not be empty");
    //     total_rows += df.num_rows();
    //     chunk_count += 1;
    // }
    //
    // assert_eq!(total_rows, expected.num_rows());
    // assert!(chunk_count > 1, "small chunk_read_limit should produce multiple chunks");

    let _ = expected;
    todo!("enable when ParquetChunkedReader is implemented");
}

/// Every chunk returned by the chunked reader has the same column names and
/// data types as the original file schema.
#[test]
#[ignore = "chunked reader not yet implemented"]
fn chunked_read_preserves_schema() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("schema.parquet");
    write_test_parquet(&path, 1000, 4);

    let expected = read_single_shot(&path);
    let expected_names = expected.column_names();

    // -- chunked read --
    // let opts = ParquetChunkedReaderOptions::builder()
    //     .path(&path)
    //     .chunk_read_limit(4096)
    //     .build()
    //     .unwrap();
    // let mut reader = ParquetChunkedReader::open(opts).unwrap();
    //
    // while let Some(df) = reader.next_chunk().unwrap() {
    //     assert_eq!(df.column_names(), expected_names,
    //         "every chunk must have the same column names as the file schema");
    //     // Verify dtypes match by checking each column's series dtype.
    //     for name in &expected_names {
    //         let expected_dt = expected.series(name).unwrap().dtype().unwrap();
    //         let actual_dt = df.series(name).unwrap().dtype().unwrap();
    //         assert_eq!(actual_dt, expected_dt,
    //             "dtype mismatch for column '{}'", name);
    //     }
    // }

    let _ = (expected_names,);
    todo!("enable when ParquetChunkedReader is implemented");
}

/// After the last chunk, `next_chunk()` returns `Ok(None)`.
#[test]
#[ignore = "chunked reader not yet implemented"]
fn chunked_read_returns_none_at_eof() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("eof.parquet");
    write_test_parquet(&path, 100, 2);

    // -- chunked read --
    // let opts = ParquetChunkedReaderOptions::builder()
    //     .path(&path)
    //     .chunk_read_limit(0) // no limit = single chunk
    //     .build()
    //     .unwrap();
    // let mut reader = ParquetChunkedReader::open(opts).unwrap();
    //
    // // Read the one chunk.
    // let chunk = reader.next_chunk().unwrap();
    // assert!(chunk.is_some(), "should produce at least one chunk");
    //
    // // Next call should be None (EOF).
    // let eof = reader.next_chunk().unwrap();
    // assert!(eof.is_none(), "should return None at EOF");

    todo!("enable when ParquetChunkedReader is implemented");
}

/// Calling `next_chunk()` after EOF keeps returning `Ok(None)` (idempotent).
#[test]
#[ignore = "chunked reader not yet implemented"]
fn chunked_read_multiple_none_at_eof() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("multi_eof.parquet");
    write_test_parquet(&path, 50, 2);

    // -- chunked read --
    // let opts = ParquetChunkedReaderOptions::builder()
    //     .path(&path)
    //     .chunk_read_limit(0)
    //     .build()
    //     .unwrap();
    // let mut reader = ParquetChunkedReader::open(opts).unwrap();
    //
    // // Drain all chunks.
    // while reader.next_chunk().unwrap().is_some() {}
    //
    // // Multiple post-EOF calls all return None.
    // for _ in 0..3 {
    //     assert!(reader.next_chunk().unwrap().is_none(),
    //         "post-EOF next_chunk() must be idempotent Ok(None)");
    // }

    todo!("enable when ParquetChunkedReader is implemented");
}

// ---------------------------------------------------------------------------
// Category 2: Chunk sizing
// ---------------------------------------------------------------------------

/// With a small `chunk_read_limit`, each chunk's device memory footprint
/// should not exceed the limit (except when a single row group is larger).
#[test]
#[ignore = "chunked reader not yet implemented"]
fn chunk_read_limit_bounds_device_memory() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("bounded.parquet");
    // 10k rows × 3 columns × 4 bytes/i32 = ~120 KiB total device data
    write_large_test_parquet(&path, 10_000);

    let limit: usize = 16 * 1024; // 16 KiB — should force many chunks

    // -- chunked read --
    // let opts = ParquetChunkedReaderOptions::builder()
    //     .path(&path)
    //     .chunk_read_limit(limit)
    //     .build()
    //     .unwrap();
    // let mut reader = ParquetChunkedReader::open(opts).unwrap();
    //
    // let mut chunk_count = 0usize;
    // while let Some(df) = reader.next_chunk().unwrap() {
    //     // Each i32 column uses 4 bytes per row, so device bytes ≈ num_rows * num_cols * 4.
    //     // Allow some overhead for null bitmasks, alignment, etc.
    //     let approx_bytes = df.num_rows() * df.num_columns() * 4;
    //     // The limit is a soft bound — a single row group may exceed it —
    //     // but the reader should not gratuitously exceed it.
    //     // We test that chunks are small, not that they are exactly at the limit.
    //     assert!(df.num_rows() > 0);
    //     chunk_count += 1;
    // }
    // assert!(chunk_count > 1, "small limit should produce multiple chunks");

    let _ = limit;
    todo!("enable when ParquetChunkedReader is implemented");
}

/// `chunk_read_limit(0)` means no limit: the reader returns everything in one
/// chunk, equivalent to a single-shot read.
#[test]
#[ignore = "chunked reader not yet implemented"]
fn chunk_read_limit_zero_means_no_limit() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("no_limit.parquet");
    write_large_test_parquet(&path, 5000);

    let expected = read_single_shot(&path);

    // -- chunked read with limit=0 --
    // let opts = ParquetChunkedReaderOptions::builder()
    //     .path(&path)
    //     .chunk_read_limit(0)
    //     .build()
    //     .unwrap();
    // let mut reader = ParquetChunkedReader::open(opts).unwrap();
    //
    // let chunk = reader.next_chunk().unwrap().expect("should return one chunk");
    // assert_eq!(chunk.num_rows(), expected.num_rows(),
    //     "limit=0 should return all rows in one chunk");
    // assert_eq!(chunk.num_columns(), expected.num_columns());
    //
    // // No more chunks.
    // assert!(reader.next_chunk().unwrap().is_none());

    let _ = expected;
    todo!("enable when ParquetChunkedReader is implemented");
}

/// Different `pass_read_limit` values change internal decompression behavior
/// but do not affect correctness — the total rows and data must be identical.
#[test]
#[ignore = "chunked reader not yet implemented"]
fn pass_read_limit_affects_decompression_not_correctness() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("pass_limit.parquet");
    write_large_test_parquet(&path, 5000);

    let expected = read_single_shot(&path);

    // -- Two reads with different pass_read_limit values --
    // for pass_limit in [0, 4096, 64 * 1024] {
    //     let opts = ParquetChunkedReaderOptions::builder()
    //         .path(&path)
    //         .chunk_read_limit(0) // no chunk limit, so one chunk
    //         .pass_read_limit(pass_limit)
    //         .build()
    //         .unwrap();
    //     let mut reader = ParquetChunkedReader::open(opts).unwrap();
    //
    //     let mut total_rows = 0usize;
    //     while let Some(df) = reader.next_chunk().unwrap() {
    //         total_rows += df.num_rows();
    //     }
    //     assert_eq!(total_rows, expected.num_rows(),
    //         "pass_read_limit={} must not change total row count", pass_limit);
    // }

    let _ = expected;
    todo!("enable when ParquetChunkedReader is implemented");
}

/// A very small `chunk_read_limit` (e.g. 1 byte) still produces valid,
/// non-empty chunks. Each chunk contains at least one row group, because a
/// row group is the minimum unit of Parquet I/O.
#[test]
#[ignore = "chunked reader not yet implemented"]
fn chunked_read_minimum_chunk_size() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("min_chunk.parquet");
    write_large_test_parquet(&path, 5000);

    let expected = read_single_shot(&path);

    // -- chunked read with absurdly small limit --
    // let opts = ParquetChunkedReaderOptions::builder()
    //     .path(&path)
    //     .chunk_read_limit(1) // 1 byte — smaller than any row group
    //     .build()
    //     .unwrap();
    // let mut reader = ParquetChunkedReader::open(opts).unwrap();
    //
    // let mut total_rows = 0usize;
    // while let Some(df) = reader.next_chunk().unwrap() {
    //     assert!(df.num_rows() > 0, "every chunk must have at least one row");
    //     assert!(df.num_columns() > 0, "every chunk must have columns");
    //     total_rows += df.num_rows();
    // }
    // assert_eq!(total_rows, expected.num_rows(),
    //     "even with 1-byte limit, all rows must be produced");

    let _ = expected;
    todo!("enable when ParquetChunkedReader is implemented");
}

/// When both `chunk_read_limit` and `pass_read_limit` are set to small
/// values, the reader still produces all rows with valid chunks.
#[test]
#[ignore = "chunked reader not yet implemented"]
fn chunked_read_combined_limits() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("combined_limits.parquet");
    write_large_test_parquet(&path, 8000);

    let expected = read_single_shot(&path);

    // -- chunked read with both limits set small --
    // let opts = ParquetChunkedReaderOptions::builder()
    //     .path(&path)
    //     .chunk_read_limit(8 * 1024)   // 8 KiB chunk limit
    //     .pass_read_limit(4 * 1024)    // 4 KiB decompression limit
    //     .build()
    //     .unwrap();
    // let mut reader = ParquetChunkedReader::open(opts).unwrap();
    //
    // let mut total_rows = 0usize;
    // let mut chunk_count = 0usize;
    // while let Some(df) = reader.next_chunk().unwrap() {
    //     assert!(df.num_rows() > 0, "chunk must not be empty");
    //     assert_eq!(df.num_columns(), expected.num_columns(),
    //         "every chunk must have all columns");
    //     total_rows += df.num_rows();
    //     chunk_count += 1;
    // }
    // assert_eq!(total_rows, expected.num_rows(),
    //     "combined limits must not lose rows");
    // assert!(chunk_count >= 1, "should produce at least one chunk");

    let _ = expected;
    todo!("enable when ParquetChunkedReader is implemented");
}

// ---------------------------------------------------------------------------
// Category 3: Column projection
// ---------------------------------------------------------------------------

/// Selecting a subset of columns produces DataFrames with only those columns.
#[test]
#[ignore = "chunked reader not yet implemented"]
fn column_projection_reduces_columns() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("projection.parquet");
    write_test_parquet(&path, 500, 5); // 5 columns: c0..c4

    // -- chunked read with column projection --
    // let opts = ParquetChunkedReaderOptions::builder()
    //     .path(&path)
    //     .columns(&["c1", "c3"])
    //     .chunk_read_limit(0)
    //     .build()
    //     .unwrap();
    // let mut reader = ParquetChunkedReader::open(opts).unwrap();
    //
    // // column_names() should reflect the projection.
    // let names = reader.column_names();
    // assert_eq!(names.len(), 2);
    // assert!(names.iter().any(|n| n == "c1"));
    // assert!(names.iter().any(|n| n == "c3"));
    //
    // let chunk = reader.next_chunk().unwrap().expect("should have data");
    // assert_eq!(chunk.num_columns(), 2);
    // assert_eq!(chunk.column_names(), vec!["c1", "c3"]);

    todo!("enable when ParquetChunkedReader is implemented");
}

/// Column projection does not change the total row count.
#[test]
#[ignore = "chunked reader not yet implemented"]
fn column_projection_preserves_row_count() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("proj_rows.parquet");
    write_test_parquet(&path, 1000, 4);

    let expected = read_single_shot(&path);

    // -- chunked read with subset of columns --
    // let opts = ParquetChunkedReaderOptions::builder()
    //     .path(&path)
    //     .columns(&["c0", "c2"])
    //     .chunk_read_limit(4096)
    //     .build()
    //     .unwrap();
    // let mut reader = ParquetChunkedReader::open(opts).unwrap();
    //
    // let mut total_rows = 0usize;
    // while let Some(df) = reader.next_chunk().unwrap() {
    //     assert_eq!(df.num_columns(), 2);
    //     total_rows += df.num_rows();
    // }
    // assert_eq!(total_rows, expected.num_rows(),
    //     "projection must not change total row count");

    let _ = expected;
    todo!("enable when ParquetChunkedReader is implemented");
}

/// Requesting a non-existent column returns an error at build/open time,
/// not silently during reading.
#[test]
#[ignore = "chunked reader not yet implemented"]
fn column_projection_invalid_column_errors() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("bad_col.parquet");
    write_test_parquet(&path, 100, 2); // columns: c0, c1

    // -- requesting non-existent column should error --
    // let opts = ParquetChunkedReaderOptions::builder()
    //     .path(&path)
    //     .columns(&["c0", "does_not_exist"])
    //     .build()
    //     .unwrap();
    //
    // let result = ParquetChunkedReader::open(opts);
    // assert!(result.is_err(),
    //     "opening with invalid column name must return an error");

    todo!("enable when ParquetChunkedReader is implemented");
}

// ---------------------------------------------------------------------------
// Category 4: Resource lifecycle
// ---------------------------------------------------------------------------

/// Reading one chunk then dropping the reader is safe. Previously returned
/// DataFrames remain valid.
#[test]
#[ignore = "chunked reader not yet implemented"]
fn dropping_reader_early_is_safe() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("early_drop.parquet");
    write_large_test_parquet(&path, 5000);

    // -- read one chunk, drop reader, use the chunk --
    // let opts = ParquetChunkedReaderOptions::builder()
    //     .path(&path)
    //     .chunk_read_limit(4096) // small limit → multiple chunks available
    //     .build()
    //     .unwrap();
    // let mut reader = ParquetChunkedReader::open(opts).unwrap();
    //
    // let first_chunk = reader.next_chunk().unwrap()
    //     .expect("should have at least one chunk");
    // let saved_rows = first_chunk.num_rows();
    //
    // // Drop reader before consuming all chunks.
    // drop(reader);
    //
    // // The previously returned DataFrame must remain valid.
    // assert_eq!(first_chunk.num_rows(), saved_rows);
    // assert!(first_chunk.num_columns() > 0);
    // // Access actual data to verify the device memory is still valid.
    // let s = first_chunk.series("c0").unwrap();
    // let _ = s.to_vec::<i32>().unwrap();

    todo!("enable when ParquetChunkedReader is implemented");
}

/// The chunked reader can be created and used inside `spawn_blocking`,
/// and the returned DataFrames can be sent out of the blocking closure.
#[test]
#[ignore = "chunked reader not yet implemented"]
fn reader_lives_in_spawn_blocking() {
    // This test verifies the runtime integration pattern from the spec.
    // We use a simple std::thread::spawn as a proxy for tokio::spawn_blocking.

    let dir = TempDir::new().unwrap();
    let path = dir.path().join("spawn_blocking.parquet");
    write_large_test_parquet(&path, 1000);

    let path_clone = path.clone();
    let handle = std::thread::spawn(move || -> Vec<DataFrame> {
        // -- create reader inside the blocking closure --
        // let opts = ParquetChunkedReaderOptions::builder()
        //     .path(&path_clone)
        //     .chunk_read_limit(0)
        //     .build()
        //     .unwrap();
        // let mut reader = ParquetChunkedReader::open(opts).unwrap();
        //
        // let mut results = Vec::new();
        // while let Some(df) = reader.next_chunk().unwrap() {
        //     results.push(df);
        // }
        // results

        let _ = path_clone;
        todo!("enable when ParquetChunkedReader is implemented");
    });

    // DataFrames cross the thread boundary (they are Send + Sync).
    let _results: Vec<DataFrame> = handle.join().unwrap();
}

/// Static assertion: the reader type must be `!Send`.
///
/// This prevents accidentally moving a reader across threads, which would
/// be unsound because the underlying C++ decoder state is not thread-safe.
#[test]
#[ignore = "chunked reader not yet implemented"]
fn reader_is_not_send() {
    // When the type exists, uncomment:
    // fn assert_not_send<T>() {
    //     // This should fail to compile if ParquetChunkedReader: Send.
    //     // We use a negative reasoning approach: if the type IS Send,
    //     // this test compiles and passes, which is wrong. So we verify
    //     // at runtime using std::marker traits.
    //     assert!(
    //         !std::any::type_name::<ParquetChunkedReader>().is_empty(),
    //         "type exists"
    //     );
    // }
    //
    // // The actual compile-time check is:
    // // static_assertions::assert_not_impl_all!(ParquetChunkedReader: Send);
    // //
    // // This requires the `static_assertions` crate or a manual approach.
    // // For now, we document the requirement. The implementation should
    // // include a compile-time assertion in the source module.

    todo!("enable when ParquetChunkedReader is implemented");
}

// ---------------------------------------------------------------------------
// Category 5: Error handling
// ---------------------------------------------------------------------------

/// Opening a non-existent file returns an error at `open()` time.
#[test]
#[ignore = "chunked reader not yet implemented"]
fn nonexistent_file_errors_at_open() {
    // let opts = ParquetChunkedReaderOptions::builder()
    //     .path("/tmp/does_not_exist_12345.parquet")
    //     .build()
    //     .unwrap();
    //
    // let result = ParquetChunkedReader::open(opts);
    // assert!(result.is_err(), "missing file must error at open()");
    // let err = result.unwrap_err();
    // assert_eq!(err.kind(), cudf::ErrorKind::IoError);

    todo!("enable when ParquetChunkedReader is implemented");
}

/// A corrupted Parquet file should produce an error at read time.
#[test]
#[ignore = "chunked reader not yet implemented"]
fn corrupted_file_errors_at_read() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("corrupt.parquet");

    // Write garbage bytes that are not a valid Parquet file.
    std::fs::write(&path, b"this is not a parquet file at all").unwrap();

    // Opening may succeed (if metadata parsing is lazy) or fail.
    // If it succeeds, the first next_chunk() must fail.
    //
    // let opts = ParquetChunkedReaderOptions::builder()
    //     .path(&path)
    //     .build()
    //     .unwrap();
    //
    // match ParquetChunkedReader::open(opts) {
    //     Err(_) => { /* acceptable: error at open */ }
    //     Ok(mut reader) => {
    //         let result = reader.next_chunk();
    //         assert!(result.is_err(),
    //             "corrupt file must error at open() or first next_chunk()");
    //     }
    // }

    todo!("enable when ParquetChunkedReader is implemented");
}

/// After `next_chunk()` returns an error, all subsequent calls return the
/// same error kind (sticky error behavior).
#[test]
#[ignore = "chunked reader not yet implemented"]
fn error_is_sticky() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("sticky.parquet");

    // Write garbage to trigger a read error.
    std::fs::write(&path, b"PAR1_but_truncated_and_invalid").unwrap();

    // let opts = ParquetChunkedReaderOptions::builder()
    //     .path(&path)
    //     .build()
    //     .unwrap();
    //
    // // This test only applies if open() succeeds with lazy metadata.
    // if let Ok(mut reader) = ParquetChunkedReader::open(opts) {
    //     let first_err = reader.next_chunk();
    //     if let Err(e1) = first_err {
    //         let kind = e1.kind();
    //
    //         // Subsequent calls must return the same error kind.
    //         for _ in 0..3 {
    //             let e = reader.next_chunk().unwrap_err();
    //             assert_eq!(e.kind(), kind,
    //                 "sticky error: subsequent errors must have the same kind");
    //         }
    //     }
    // }

    todo!("enable when ParquetChunkedReader is implemented");
}

// ---------------------------------------------------------------------------
// Category 6: Metadata
// ---------------------------------------------------------------------------

/// Column names are available from metadata after `open()`, without reading
/// any data chunks.
#[test]
#[ignore = "chunked reader not yet implemented"]
fn metadata_available_without_reading() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("metadata.parquet");
    write_test_parquet(&path, 100, 3); // columns: c0, c1, c2

    // -- open reader and inspect metadata without reading --
    // let opts = ParquetChunkedReaderOptions::builder()
    //     .path(&path)
    //     .build()
    //     .unwrap();
    // let reader = ParquetChunkedReader::open(opts).unwrap();
    //
    // let names = reader.column_names();
    // assert_eq!(names.len(), 3);
    // assert_eq!(names[0], "c0");
    // assert_eq!(names[1], "c1");
    // assert_eq!(names[2], "c2");
    //
    // // Reader can be dropped without reading any data.
    // drop(reader);

    todo!("enable when ParquetChunkedReader is implemented");
}

/// Column names from metadata match the column names in each data chunk.
#[test]
#[ignore = "chunked reader not yet implemented"]
fn metadata_column_names_match_data() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("meta_match.parquet");
    write_test_parquet(&path, 500, 3);

    // -- verify metadata names match chunk names --
    // let opts = ParquetChunkedReaderOptions::builder()
    //     .path(&path)
    //     .chunk_read_limit(4096)
    //     .build()
    //     .unwrap();
    // let mut reader = ParquetChunkedReader::open(opts).unwrap();
    //
    // let meta_names: Vec<String> = reader.column_names().to_vec();
    //
    // while let Some(df) = reader.next_chunk().unwrap() {
    //     let chunk_names: Vec<&str> = df.column_names();
    //     assert_eq!(chunk_names.len(), meta_names.len());
    //     for (cn, mn) in chunk_names.iter().zip(meta_names.iter()) {
    //         assert_eq!(*cn, mn.as_str(),
    //             "chunk column names must match metadata column names");
    //     }
    // }

    todo!("enable when ParquetChunkedReader is implemented");
}

// ---------------------------------------------------------------------------
// Category 7: Data integrity
// ---------------------------------------------------------------------------

/// Chunked reads must produce byte-identical column data compared to a
/// single-shot read of the same file. This is the fundamental correctness
/// property: chunking is a performance mechanism, not a semantic change.
///
/// The test writes a Parquet file, reads it single-shot, then reads it
/// chunked with a small limit. It collects all per-column values from
/// the chunked read (in order) and asserts they match the single-shot values.
#[test]
#[ignore = "chunked reader not yet implemented"]
fn chunked_read_data_matches_single_shot() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("data_integrity.parquet");
    write_test_parquet(&path, 6000, 3); // 3 columns: c0, c1, c2

    let expected = read_single_shot(&path);

    // -- chunked read and collect all column values --
    // let opts = ParquetChunkedReaderOptions::builder()
    //     .path(&path)
    //     .chunk_read_limit(4096) // small limit to force multiple chunks
    //     .build()
    //     .unwrap();
    // let mut reader = ParquetChunkedReader::open(opts).unwrap();
    //
    // // Accumulate per-column values across all chunks.
    // let num_cols = expected.num_columns();
    // let col_names = expected.column_names();
    // let mut accumulated: Vec<Vec<i32>> = vec![Vec::new(); num_cols];
    //
    // while let Some(df) = reader.next_chunk().unwrap() {
    //     for (i, name) in col_names.iter().enumerate() {
    //         let chunk_values = df.series(name).unwrap().to_vec::<i32>().unwrap();
    //         accumulated[i].extend_from_slice(&chunk_values);
    //     }
    // }
    //
    // // Compare against single-shot read values.
    // for (i, name) in col_names.iter().enumerate() {
    //     let expected_values = expected.series(name).unwrap().to_vec::<i32>().unwrap();
    //     assert_eq!(accumulated[i].len(), expected_values.len(),
    //         "column '{}': row count mismatch", name);
    //     assert_eq!(accumulated[i], expected_values,
    //         "column '{}': data mismatch between chunked and single-shot read", name);
    // }

    let _ = expected;
    todo!("enable when ParquetChunkedReader is implemented");
}
