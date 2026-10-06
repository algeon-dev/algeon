//! Data-only memory-control and source-chunk planning contracts.
//!
//! Planning sizes source chunks from these policies; the runtime enforces the
//! device budget they carry.

use crate::error::{QueryError, QueryErrorCode, Result};

/// The device budget and source chunking one execution runs under. An admitted
/// attempt's budget is its grant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MemoryControlPolicy {
    pub device_budget_bytes: u64,
    /// Source-byte grouping target. `u64::MAX` leaves the configured target
    /// unset; source tasks derive a finite target from the admitted grant.
    pub max_source_chunk_bytes: u64,
    pub max_row_groups_per_chunk: usize,
    pub max_retry_attempts: usize,
    pub min_retry_chunk_row_groups: usize,
}

impl MemoryControlPolicy {
    /// No configured source-byte ceiling; the immutable grant sizes batches.
    pub const DEFAULT_MAX_SOURCE_CHUNK_BYTES: u64 = u64::MAX;
    /// Individual row groups remain retryable units within coalesced reads.
    pub const DEFAULT_MAX_ROW_GROUPS_PER_CHUNK: usize = 1;
    pub const DEFAULT_MAX_RETRY_ATTEMPTS: usize = 3;
    pub const DEFAULT_MIN_RETRY_CHUNK_ROW_GROUPS: usize = 1;

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

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SourceChunkPlanningMode {
    #[default]
    Conservative,
    LocalParquetThroughput,
}

impl SourceChunkPlanningMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Conservative => "conservative",
            Self::LocalParquetThroughput => "local_parquet_throughput",
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
