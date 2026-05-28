//! GPU-accelerated row hashing.
//!
//! Computes hash values for each row in a table. Useful for hash-based
//! partitioning, deduplication, and fingerprinting.
//!
//! # Example
//!
//! ```no_run
//! use cudf::column::Column;
//! use cudf::table::Table;
//! use cudf::hashing;
//!
//! let col = Column::from_slice(&[10i32, 20, 30]).unwrap();
//! let table = Table::from_columns(vec![col]).unwrap();
//! let view = table.view().unwrap();
//!
//! let hashes = hashing::murmurhash3_x86_32(&view, 0).unwrap();
//! // hashes is a UINT32 column with one hash per row.
//! ```

use crate::column::Column;
use crate::error::{self, Result};
use crate::table::TableView;
use std::ptr;

/// Default hash seed (matches cudf::hashing::DEFAULT_HASH_SEED).
pub const DEFAULT_HASH_SEED: u32 = 0;

/// Compute MurmurHash3 32-bit hash of each row in a table.
///
/// Returns a `UINT32` column with one hash value per row. The seed is used
/// for the first column; subsequent columns chain the previous hash as seed.
pub fn murmurhash3_x86_32(input: &TableView<'_>, seed: u32) -> Result<Column> {
    let mut out = ptr::null_mut();
    error::check_status_in_with(
        unsafe { cudf_sys::cudf_hashing_murmurhash3_x86_32(input.raw_handle(), seed, &mut out) },
        "hashing::murmurhash3_x86_32",
        [("seed", seed.to_string())],
    )?;
    Ok(unsafe { Column::from_raw(out) })
}

/// Compute XXHash 64-bit hash of each row in a table.
///
/// Returns a `UINT64` column with one hash value per row.
pub fn xxhash_64(input: &TableView<'_>, seed: u64) -> Result<Column> {
    let mut out = ptr::null_mut();
    error::check_status_in_with(
        unsafe { cudf_sys::cudf_hashing_xxhash_64(input.raw_handle(), seed, &mut out) },
        "hashing::xxhash_64",
        [("seed", seed.to_string())],
    )?;
    Ok(unsafe { Column::from_raw(out) })
}
