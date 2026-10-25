//! Data-only memory-control and source-chunk planning contracts.
//!
//! Planning sizes source chunks from these policies; the runtime enforces the
//! device budget they carry.

use crate::error::{QueryError, QueryErrorCode, Result};

/// The device budget and source chunking one execution runs under. An admitted
/// attempt's budget is its grant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MemoryControlPolicy {
    /// Allocation cap of the attempt. The runtime's device ledger enforces it;
    /// planning only sizes chunks against it.
    pub device_budget_bytes: u64,
    /// Source-byte grouping target, compared with each chunk's decoded size
    /// estimate (`crate::chunk::decoded`). `u64::MAX` leaves the configured
    /// target unset; source tasks derive a finite target from the admitted grant.
    pub max_source_chunk_bytes: u64,
    /// Row groups coalesced into one source chunk. The throughput planning
    /// mode may raise it to the row-group cap `crate::chunk` owns.
    pub max_row_groups_per_chunk: usize,
    /// Attempts the engine's retry may spend splitting one chunk after an
    /// allocation failure.
    pub max_retry_attempts: usize,
    /// Smallest chunk, in row groups, a retry may split down to.
    pub min_retry_chunk_row_groups: usize,
}

impl MemoryControlPolicy {
    /// No configured source-byte ceiling; the immutable grant sizes batches.
    pub const DEFAULT_MAX_SOURCE_CHUNK_BYTES: u64 = u64::MAX;
    /// One row group per chunk. The default throughput planning mode raises
    /// this cap for local Parquet sources; the conservative mode keeps it.
    pub const DEFAULT_MAX_ROW_GROUPS_PER_CHUNK: usize = 1;
    /// Default retry limit for splitting a source chunk after allocation failure.
    pub const DEFAULT_MAX_RETRY_ATTEMPTS: usize = 3;
    /// Default minimum retry chunk of one row group.
    pub const DEFAULT_MIN_RETRY_CHUNK_ROW_GROUPS: usize = 1;

    /// Rejects a zero in any field; the error names the field in its `field`
    /// fact. Zero is not a "disabled" value here, since each field bounds a
    /// loop or a division in chunk planning.
    pub fn validate(self) -> Result<()> {
        let Self {
            device_budget_bytes,
            max_source_chunk_bytes,
            max_row_groups_per_chunk,
            max_retry_attempts,
            min_retry_chunk_row_groups,
        } = self;
        validate_nonzero_policy_field("device_budget_bytes", device_budget_bytes == 0)?;
        validate_nonzero_policy_field("max_source_chunk_bytes", max_source_chunk_bytes == 0)?;
        validate_nonzero_policy_field("max_row_groups_per_chunk", max_row_groups_per_chunk == 0)?;
        validate_nonzero_policy_field("max_retry_attempts", max_retry_attempts == 0)?;
        validate_nonzero_policy_field("min_retry_chunk_row_groups", min_retry_chunk_row_groups == 0)
    }
}

/// How source chunk planning sizes local Parquet chunks. The chosen mode and
/// the reason for it are recorded in each chunk plan.
///
/// In both modes the source-byte target bounds each chunk's decoded size, and
/// a chunk whose allocation the query cap rejects is split by row group and
/// retried within the policy's retry limits.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SourceChunkPlanningMode {
    /// Chunks follow `max_row_groups_per_chunk`, one row group by default.
    /// Each chunk is then the smallest retry unit, but per-chunk work such as
    /// partial aggregation grows with the table's row-group count.
    Conservative,
    /// Raises a one-row-group cap so a chunk of a Parquet source, local or
    /// remote, can hold many row groups up to the source-byte target;
    /// `crate::chunk` owns the row-group cap and the source conditions. A
    /// source that does not qualify keeps conservative chunks.
    #[default]
    ParquetThroughput,
}

impl SourceChunkPlanningMode {
    /// Stable source-planning mode label used in reports and configuration.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Conservative => "conservative",
            Self::ParquetThroughput => "parquet_throughput",
        }
    }
}

impl std::fmt::Display for SourceChunkPlanningMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

fn validate_nonzero_policy_field(field: &'static str, is_zero: bool) -> Result<()> {
    if is_zero {
        return Err(QueryError::new(
            QueryErrorCode::InvalidConfiguration,
            "query_engine.memory_policy.validate",
            "bounded memory-control policy fields must be greater than zero",
        )
        .with_fact("policy", "bounded")
        .with_fact("field", field)
        .with_fact("reason", "zero"));
    }
    Ok(())
}
