//! GPU-accelerated row hashing.
//!
//! Computes hash values for each row in a table. Useful for hash-based
//! partitioning, deduplication, and fingerprinting.
//!
//! # Example
//!
//! ```no_run
//! use cudf::{
//!     core::{Column, Table},
//!     ops::hashing,
//! };
//!
//! let col = Column::from_slice(&[10i32, 20, 30]).unwrap();
//! let table = Table::from_columns(vec![col]).unwrap();
//! let view = table.view().unwrap();
//!
//! let hashes = hashing::murmurhash3_x86_32(&view, 0).unwrap();
//! // hashes is a UINT32 column with one hash per row.
//! ```

use std::ptr;

use crate::core::{
    column::Column,
    error::{self, Result},
    table::TableView,
};

/// Default hash seed (kept aligned with the legacy root hashing constant).
pub const DEFAULT_HASH_SEED: u32 = 0;

fn murmurhash3_x86_32_impl(
    input: &TableView<'_>,
    seed: u32,
    exec_options: Option<&cudf_sys::cudf_exec_options_t>,
    operation: &'static str,
) -> Result<Column> {
    let mut out = ptr::null_mut();
    let status = match exec_options {
        Some(options) => unsafe {
            cudf_sys::cudf_hashing_murmurhash3_x86_32_ex(input.raw_handle(), seed, &mut out, options)
        },
        None => unsafe { cudf_sys::cudf_hashing_murmurhash3_x86_32(input.raw_handle(), seed, &mut out) },
    };
    error::check_status_in_with(
        status,
        operation,
        [("seed", seed.to_string()), ("execution_context_provided", exec_options.is_some().to_string())],
    )?;
    Ok(unsafe { Column::from_raw(out) })
}

fn xxhash_64_impl(
    input: &TableView<'_>,
    seed: u64,
    exec_options: Option<&cudf_sys::cudf_exec_options_t>,
    operation: &'static str,
) -> Result<Column> {
    let mut out = ptr::null_mut();
    let status = match exec_options {
        Some(options) => unsafe { cudf_sys::cudf_hashing_xxhash_64_ex(input.raw_handle(), seed, &mut out, options) },
        None => unsafe { cudf_sys::cudf_hashing_xxhash_64(input.raw_handle(), seed, &mut out) },
    };
    error::check_status_in_with(
        status,
        operation,
        [("seed", seed.to_string()), ("execution_context_provided", exec_options.is_some().to_string())],
    )?;
    Ok(unsafe { Column::from_raw(out) })
}

/// Compute MurmurHash3 32-bit hash of each row in a table.
///
/// Returns a `UINT32` column with one hash value per row. The seed is used
/// for the first column; subsequent columns chain the previous hash as seed.
pub fn murmurhash3_x86_32(input: &TableView<'_>, seed: u32) -> Result<Column> {
    let output_domain = std::sync::Arc::clone(input.execution_domain());
    let exec_options = output_domain.explicit_exec_options_raw();
    murmurhash3_x86_32_impl(input, seed, exec_options.as_ref(), "hashing::murmurhash3_x86_32")
        .map(|column| column.with_execution_domain(output_domain))
}

/// Compute XXHash 64-bit hash of each row in a table.
///
/// Returns a `UINT64` column with one hash value per row.
pub fn xxhash_64(input: &TableView<'_>, seed: u64) -> Result<Column> {
    let output_domain = std::sync::Arc::clone(input.execution_domain());
    let exec_options = output_domain.explicit_exec_options_raw();
    xxhash_64_impl(input, seed, exec_options.as_ref(), "hashing::xxhash_64")
        .map(|column| column.with_execution_domain(output_domain))
}
