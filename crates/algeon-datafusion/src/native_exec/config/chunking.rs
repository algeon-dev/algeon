//! Source chunking settings of native reads.

use std::num::{NonZeroU64, NonZeroUsize};

use query_planning::execution_contracts::MemoryControlPolicy;

use crate::error::{Error, ErrorCode, Result};

/// Source chunking of native reads. Device capacity always comes from the
/// admitted grant, so a chunking carries no device budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourceChunking {
    /// Source bytes planned into one chunk; a planning target.
    /// `u64::MAX` derives the target from the immutable admitted grant.
    pub max_source_chunk_bytes: NonZeroU64,
    /// Row groups planned into one chunk.
    pub max_row_groups_per_chunk: NonZeroUsize,
    /// Attempts a failed chunk read retries with smaller chunks.
    pub max_retry_attempts: NonZeroUsize,
    /// Row-group floor of a retried chunk.
    pub min_retry_chunk_row_groups: NonZeroUsize,
}

impl SourceChunking {
    /// Default source-byte setting, without a configured ceiling.
    pub const DEFAULT_MAX_SOURCE_CHUNK_BYTES: NonZeroU64 =
        NonZeroU64::new(MemoryControlPolicy::DEFAULT_MAX_SOURCE_CHUNK_BYTES).unwrap();
    /// Default row groups planned into one chunk.
    pub const DEFAULT_MAX_ROW_GROUPS_PER_CHUNK: NonZeroUsize =
        NonZeroUsize::new(MemoryControlPolicy::DEFAULT_MAX_ROW_GROUPS_PER_CHUNK).unwrap();
    /// Default retry attempts of a failed chunk read.
    pub const DEFAULT_MAX_RETRY_ATTEMPTS: NonZeroUsize =
        NonZeroUsize::new(MemoryControlPolicy::DEFAULT_MAX_RETRY_ATTEMPTS).unwrap();
    /// Default row-group floor of a retried chunk.
    pub const DEFAULT_MIN_RETRY_CHUNK_ROW_GROUPS: NonZeroUsize =
        NonZeroUsize::new(MemoryControlPolicy::DEFAULT_MIN_RETRY_CHUNK_ROW_GROUPS).unwrap();

    /// Chunking from plain counts; each must be greater than zero.
    pub fn try_new(
        max_source_chunk_bytes: u64,
        max_row_groups_per_chunk: usize,
        max_retry_attempts: usize,
        min_retry_chunk_row_groups: usize,
    ) -> Result<Self> {
        Ok(Self {
            max_source_chunk_bytes: NonZeroU64::new(max_source_chunk_bytes)
                .ok_or_else(|| zero_source_chunking_field("max_source_chunk_bytes"))?,
            max_row_groups_per_chunk: NonZeroUsize::new(max_row_groups_per_chunk)
                .ok_or_else(|| zero_source_chunking_field("max_row_groups_per_chunk"))?,
            max_retry_attempts: NonZeroUsize::new(max_retry_attempts)
                .ok_or_else(|| zero_source_chunking_field("max_retry_attempts"))?,
            min_retry_chunk_row_groups: NonZeroUsize::new(min_retry_chunk_row_groups)
                .ok_or_else(|| zero_source_chunking_field("min_retry_chunk_row_groups"))?,
        })
    }
}

impl Default for SourceChunking {
    fn default() -> Self {
        Self {
            max_source_chunk_bytes: Self::DEFAULT_MAX_SOURCE_CHUNK_BYTES,
            max_row_groups_per_chunk: Self::DEFAULT_MAX_ROW_GROUPS_PER_CHUNK,
            max_retry_attempts: Self::DEFAULT_MAX_RETRY_ATTEMPTS,
            min_retry_chunk_row_groups: Self::DEFAULT_MIN_RETRY_CHUNK_ROW_GROUPS,
        }
    }
}

fn zero_source_chunking_field(field: &'static str) -> Error {
    Error::new(ErrorCode::InvalidRequest, "algeon_datafusion.native_exec.source_chunking", "must be greater than zero")
        .with_fact("field", field)
}
