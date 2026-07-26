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
    runtime::{
        execution_context::ExecutionOverrides, execution_domain::ExecutionDomain, execution_resolver::ResolvedExecution,
    },
};

/// Default hash seed (kept aligned with the legacy root hashing constant).
pub const DEFAULT_HASH_SEED: u32 = 0;

fn murmurhash3_x86_32_impl(
    input: &TableView<'_>,
    seed: u32,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: &cudf_sys::cudf_exec_options_t,
    operation: &'static str,
) -> Result<Column> {
    let mut out = ptr::null_mut();
    let status =
        unsafe { cudf_sys::cudf_hashing_murmurhash3_x86_32_ex(input.raw_handle(), seed, &mut out, exec_options) };
    error::check_status_in_with_details(
        status,
        operation,
        [("seed", seed.to_string()), ("execution_authority", "retained_domain".to_string())],
    )?;
    unsafe { Column::from_raw_with_domain(out, Arc::clone(output_domain)) }
}

fn xxhash_64_impl(
    input: &TableView<'_>,
    seed: u64,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: &cudf_sys::cudf_exec_options_t,
    operation: &'static str,
) -> Result<Column> {
    let mut out = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_hashing_xxhash_64_ex(input.raw_handle(), seed, &mut out, exec_options) };
    error::check_status_in_with_details(
        status,
        operation,
        [("seed", seed.to_string()), ("execution_authority", "retained_domain".to_string())],
    )?;
    unsafe { Column::from_raw_with_domain(out, Arc::clone(output_domain)) }
}

/// Compute MurmurHash3 32-bit hash of each row in a table.
///
/// Returns a `UINT32` column with one hash value per row. The seed is used
/// for the first column; subsequent columns chain the previous hash as seed.
pub fn murmurhash3_x86_32(input: &TableView<'_>, seed: u32) -> Result<Column> {
    let ctx = ExecutionOverrides::new();
    let execution = ResolvedExecution::unary(input.execution_domain(), &ctx, "hashing::murmurhash3_x86_32")?;
    murmurhash3_x86_32_impl(
        input,
        seed,
        execution.output_domain(),
        execution.exec_options(),
        "hashing::murmurhash3_x86_32",
    )
}

/// Compute XXHash 64-bit hash of each row in a table.
///
/// Returns a `UINT64` column with one hash value per row.
pub fn xxhash_64(input: &TableView<'_>, seed: u64) -> Result<Column> {
    let ctx = ExecutionOverrides::new();
    let execution = ResolvedExecution::unary(input.execution_domain(), &ctx, "hashing::xxhash_64")?;
    xxhash_64_impl(input, seed, execution.output_domain(), execution.exec_options(), "hashing::xxhash_64")
}
