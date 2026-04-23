//! GPU-accelerated row hashing.
//!
//! Computes hash values for each row in a table. Useful for hash-based
//! partitioning, deduplication, and fingerprinting.
//!
//! This module currently exposes low-level `TableView`-based primitives.
//! Most users should stay on the higher-level `DataFrame` workflows unless
//! they explicitly need row-hash columns from the core surface.

use std::{ptr, sync::Arc};

use crate::{
    core::{
        column::Column,
        error::{self, Result},
        table::TableView,
    },
    runtime::execution_domain::ExecutionDomain,
};

/// Default hash seed (kept aligned with the legacy root hashing constant).
pub const DEFAULT_HASH_SEED: u32 = 0;

fn murmurhash3_x86_32_impl(
    input: &TableView<'_>,
    seed: u32,
    output_domain: &Arc<ExecutionDomain>,
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
    error::check_status_in_with_details(
        status,
        operation,
        [("seed", seed.to_string()), ("execution_context_provided", exec_options.is_some().to_string())],
    )?;
    unsafe { Column::from_raw_with_domain(out, Arc::clone(output_domain)) }
}

fn xxhash_64_impl(
    input: &TableView<'_>,
    seed: u64,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: Option<&cudf_sys::cudf_exec_options_t>,
    operation: &'static str,
) -> Result<Column> {
    let mut out = ptr::null_mut();
    let status = match exec_options {
        Some(options) => unsafe { cudf_sys::cudf_hashing_xxhash_64_ex(input.raw_handle(), seed, &mut out, options) },
        None => unsafe { cudf_sys::cudf_hashing_xxhash_64(input.raw_handle(), seed, &mut out) },
    };
    error::check_status_in_with_details(
        status,
        operation,
        [("seed", seed.to_string()), ("execution_context_provided", exec_options.is_some().to_string())],
    )?;
    unsafe { Column::from_raw_with_domain(out, Arc::clone(output_domain)) }
}

/// Compute MurmurHash3 32-bit hash of each row in a table.
///
/// Returns a `UINT32` column with one hash value per row. The seed is used
/// for the first column; subsequent columns chain the previous hash as seed.
pub fn murmurhash3_x86_32(input: &TableView<'_>, seed: u32) -> Result<Column> {
    let output_domain = Arc::clone(input.execution_domain());
    let exec_options = output_domain.explicit_exec_options_raw();
    murmurhash3_x86_32_impl(input, seed, &output_domain, exec_options.as_ref(), "hashing::murmurhash3_x86_32")
}

/// Compute XXHash 64-bit hash of each row in a table.
///
/// Returns a `UINT64` column with one hash value per row.
pub fn xxhash_64(input: &TableView<'_>, seed: u64) -> Result<Column> {
    let output_domain = Arc::clone(input.execution_domain());
    let exec_options = output_domain.explicit_exec_options_raw();
    xxhash_64_impl(input, seed, &output_domain, exec_options.as_ref(), "hashing::xxhash_64")
}
