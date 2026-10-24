use super::memory::SourceTaskMemoryModel;
use crate::{
    error::{QueryError, QueryErrorCode, Result as QueryResult},
    execution_contracts::{MemoryControlPolicy, SourceChunkPlanningMode},
    source::{LocalParquetFile, LocalParquetSource, ParquetReadStrategy},
};

/// Row-group cap used by throughput source planning when the configured
/// policy is the conservative one-row-group shape. It bounds how much source
/// one chunk retry reads again; the source-byte target sizes the chunk.
pub const PARQUET_THROUGHPUT_MAX_ROW_GROUPS_PER_CHUNK: usize =
    SourceChunkPlanningOptions::DEFAULT_PARQUET_THROUGHPUT_MAX_ROW_GROUPS_PER_CHUNK;

/// Expert policy for Parquet source chunk planning heuristics.
///
/// This controls when [`SourceChunkPlanningMode::ParquetThroughput`] may
/// widen conservative one-row-group chunks. It is intentionally a Rust-only
/// policy until production evidence shows a stable tuning model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct SourceChunkPlanningOptions {
    parquet_throughput_max_row_groups_per_chunk: usize,
}

impl SourceChunkPlanningOptions {
    pub(crate) const DEFAULT_PARQUET_THROUGHPUT_MAX_ROW_GROUPS_PER_CHUNK: usize = 1024;

    pub(crate) fn parquet_throughput_max_row_groups_per_chunk(&self) -> usize {
        self.parquet_throughput_max_row_groups_per_chunk
    }

    /// Sets the row-group cap available when throughput planning widens a
    /// conservative chunk. The value must be nonzero; call [`Self::validate`]
    /// after changing options.
    pub fn with_parquet_throughput_max_row_groups_per_chunk(mut self, row_groups: usize) -> Self {
        self.parquet_throughput_max_row_groups_per_chunk = row_groups;
        self
    }

    /// Rejects a zero row-group limit with [`QueryErrorCode::InvalidConfiguration`].
    pub fn validate(&self) -> QueryResult<()> {
        validate_source_chunk_planning_nonzero_usize(
            "parquet_throughput_max_row_groups_per_chunk",
            self.parquet_throughput_max_row_groups_per_chunk,
        )
    }
}

impl Default for SourceChunkPlanningOptions {
    fn default() -> Self {
        Self { parquet_throughput_max_row_groups_per_chunk: Self::DEFAULT_PARQUET_THROUGHPUT_MAX_ROW_GROUPS_PER_CHUNK }
    }
}

fn validate_source_chunk_planning_nonzero_usize(field: &'static str, value: usize) -> QueryResult<()> {
    if value == 0 {
        return Err(source_chunk_planning_zero_error(field, value));
    }
    Ok(())
}

fn source_chunk_planning_zero_error(field: &'static str, value: impl std::fmt::Display) -> QueryError {
    QueryError::new(
        QueryErrorCode::InvalidConfiguration,
        "query_engine.source_chunk_planning_options.validate",
        "source chunk planning option must be greater than zero",
    )
    .with_fact("field", field)
    .with_fact("value", value)
    .with_fact("reason", "zero")
}

/// Policy that controls how a local Parquet source is split into chunks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourceChunkPlanningPolicy {
    /// Target upper bound, in decoded bytes ([`crate::chunk::decoded`]), for
    /// one chunk's read columns. A row group without a decoded estimate counts
    /// its encoded source-read bytes.
    ///
    /// `None` means row-group grouping is constrained only by
    /// [`max_row_groups_per_chunk`](Self::max_row_groups_per_chunk).
    pub max_source_chunk_bytes: Option<u64>,
    /// Maximum number of row groups assigned to one chunk.
    ///
    /// Callers must pass a value greater than zero; unbounded policies use
    /// `usize::MAX`.
    pub max_row_groups_per_chunk: usize,
    source_chunk_planning_mode: SourceChunkPlanningMode,
    source_chunk_planning_reason: &'static str,
}

impl SourceChunkPlanningPolicy {
    pub const fn new(max_source_chunk_bytes: Option<u64>, max_row_groups_per_chunk: usize) -> Self {
        Self {
            max_source_chunk_bytes,
            max_row_groups_per_chunk,
            source_chunk_planning_mode: SourceChunkPlanningMode::Conservative,
            source_chunk_planning_reason: "explicit_policy",
        }
    }

    pub(crate) fn source_chunk_planning_mode(self) -> SourceChunkPlanningMode {
        self.source_chunk_planning_mode
    }

    pub fn source_chunk_planning_reason(self) -> &'static str {
        self.source_chunk_planning_reason
    }

    pub fn mark_downstream_cap(&mut self, reason: &'static str) {
        self.source_chunk_planning_reason = reason;
    }

    /// Derives a source-task policy from the bounded memory policy and slot model.
    ///
    /// The default source-byte setting uses
    /// `device_budget_bytes / model.live_slot_count()`. Explicit targets at or
    /// below the grant are bounded by the same live-slot target. An explicit
    /// target above the grant remains available for allocation retry. Both
    /// targets bound the same decoded estimate.
    pub fn for_source_task(memory_policy: MemoryControlPolicy, model: SourceTaskMemoryModel) -> Self {
        let base = Self::from(memory_policy);
        let MemoryControlPolicy { device_budget_bytes, max_source_chunk_bytes, max_row_groups_per_chunk, .. } =
            memory_policy;
        if max_source_chunk_bytes > device_budget_bytes
            && max_source_chunk_bytes != MemoryControlPolicy::DEFAULT_MAX_SOURCE_CHUNK_BYTES
        {
            return base;
        }
        let budget_target = device_budget_bytes / model.live_slot_count().max(1);
        let task_target = max_source_chunk_bytes.min(budget_target.max(1));
        let mut policy = Self::new(Some(task_target), max_row_groups_per_chunk);
        policy.source_chunk_planning_reason =
            if task_target < max_source_chunk_bytes { "memory_cap" } else { "configured_policy" };
        policy
    }

    /// Derives a source-task policy with explicit throughput planning options.
    pub fn for_source_task_with_mode_and_options(
        source: &LocalParquetSource,
        memory_policy: MemoryControlPolicy,
        model: SourceTaskMemoryModel,
        mode: SourceChunkPlanningMode,
        options: SourceChunkPlanningOptions,
    ) -> Self {
        Self::for_source_task(memory_policy, model).with_source_planning_mode(source, mode, options)
    }

    /// Derives a source policy from memory policy and source planning mode.
    pub fn from_memory_policy_with_mode(
        source: &LocalParquetSource,
        memory_policy: MemoryControlPolicy,
        mode: SourceChunkPlanningMode,
    ) -> Self {
        Self::from_memory_policy_with_mode_and_options(
            source,
            memory_policy,
            mode,
            SourceChunkPlanningOptions::default(),
        )
    }

    /// Derives a source policy with explicit throughput planning options.
    pub fn from_memory_policy_with_mode_and_options(
        source: &LocalParquetSource,
        memory_policy: MemoryControlPolicy,
        mode: SourceChunkPlanningMode,
        options: SourceChunkPlanningOptions,
    ) -> Self {
        Self::from(memory_policy).with_source_planning_mode(source, mode, options)
    }

    pub(super) fn effective_max_row_groups_per_chunk(self) -> usize {
        self.max_row_groups_per_chunk.max(1)
    }

    fn with_source_planning_mode(
        mut self,
        source: &LocalParquetSource,
        mode: SourceChunkPlanningMode,
        options: SourceChunkPlanningOptions,
    ) -> Self {
        if mode != SourceChunkPlanningMode::ParquetThroughput {
            return self;
        }
        self.source_chunk_planning_mode = mode;
        if !parquet_throughput_planning_admitted(source, self) {
            self.source_chunk_planning_reason = "parquet_throughput_rejected";
            return self;
        }
        self.max_row_groups_per_chunk =
            self.max_row_groups_per_chunk.max(options.parquet_throughput_max_row_groups_per_chunk());
        self.source_chunk_planning_reason = "parquet_throughput_admitted";
        self
    }
}

impl Default for SourceChunkPlanningPolicy {
    fn default() -> Self {
        Self {
            max_source_chunk_bytes: None,
            max_row_groups_per_chunk: usize::MAX,
            source_chunk_planning_mode: SourceChunkPlanningMode::Conservative,
            source_chunk_planning_reason: "unbounded_policy",
        }
    }
}

impl From<MemoryControlPolicy> for SourceChunkPlanningPolicy {
    fn from(memory_policy: MemoryControlPolicy) -> Self {
        let target = if memory_policy.max_source_chunk_bytes == MemoryControlPolicy::DEFAULT_MAX_SOURCE_CHUNK_BYTES {
            (memory_policy.device_budget_bytes / SourceTaskMemoryModel::aggregate_pipeline(false).live_slot_count())
                .max(1)
        } else {
            memory_policy.max_source_chunk_bytes
        };
        let mut policy = Self::new(Some(target), memory_policy.max_row_groups_per_chunk);
        policy.source_chunk_planning_reason =
            if memory_policy.max_source_chunk_bytes == MemoryControlPolicy::DEFAULT_MAX_SOURCE_CHUNK_BYTES {
                "memory_cap"
            } else {
                "configured_policy"
            };
        policy
    }
}

/// Whether throughput planning may raise a one-row-group cap for `source`.
/// The grant does not enter here: the source-byte target already scales the
/// widened chunks with it.
fn parquet_throughput_planning_admitted(source: &LocalParquetSource, policy: SourceChunkPlanningPolicy) -> bool {
    if policy.max_row_groups_per_chunk != 1 || policy.max_source_chunk_bytes.is_none() {
        return false;
    }
    if source.files.is_empty()
        || source.is_zero_column_count_source()
        || source.scan_limit.is_some()
        || source.read_columns.is_empty()
        || !source.partition_columns.is_empty()
    {
        return false;
    }
    source.files.iter().all(|file| {
        parquet_throughput_file_uses_supported_reader_path(file)
            && file.partition_values.is_empty()
            && parquet_throughput_file_has_source_read_bytes(source, file)
    })
}

/// A remote file qualifies like a local one: one chunk is one reader call
/// over its row groups, and the reader fetches their byte ranges in parallel.
/// Both need the row-group byte metadata that planning reads from the footer.
fn parquet_throughput_file_uses_supported_reader_path(file: &LocalParquetFile) -> bool {
    match file.resolved_source.read_strategy() {
        ParquetReadStrategy::LocalKvikio => file.location.is_local_path(),
        ParquetReadStrategy::LocalCacheKvikio => file.path.is_file(),
        ParquetReadStrategy::RemoteDirect => !file.location.is_local_path(),
    }
}

fn parquet_throughput_file_has_source_read_bytes(source: &LocalParquetSource, file: &LocalParquetFile) -> bool {
    let selected_row_groups = match &file.row_groups {
        Some(row_groups) if !row_groups.is_empty() => row_groups.clone(),
        Some(_) => return false,
        None => {
            let Some(count) = file.projected_row_group_uncompressed_bytes.as_ref().map(Vec::len) else {
                return false;
            };
            if count == 0 {
                return false;
            }
            (0..count).collect()
        },
    };
    let Some(max_row_group) = selected_row_groups.iter().copied().max() else {
        return false;
    };
    let projected_covers_selection =
        file.projected_row_group_uncompressed_bytes.as_ref().is_some_and(|bytes| max_row_group < bytes.len());
    let pure_filter_covers_selection = source.pure_filter_columns.is_empty()
        || file.pure_filter_row_group_uncompressed_bytes.as_ref().is_some_and(|bytes| max_row_group < bytes.len());
    projected_covers_selection && pure_filter_covers_selection
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::datatypes::{DataType, Field, Schema};

    use super::*;
    use crate::source::{CredentialMode, ResolvedParquetSource};

    fn one_row_group_policy() -> MemoryControlPolicy {
        MemoryControlPolicy {
            device_budget_bytes: 64 * 1024 * 1024 * 1024,
            max_source_chunk_bytes: 64 * 1024 * 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 3,
            min_retry_chunk_row_groups: 1,
        }
    }

    fn remote_source(file: LocalParquetFile) -> LocalParquetSource {
        let schema = Arc::new(Schema::new(vec![Field::new("c0", DataType::Int64, true)]));
        LocalParquetSource::from_files(vec![file], schema, vec!["c0".to_string()])
    }

    fn remote_file() -> LocalParquetFile {
        let resolved =
            ResolvedParquetSource::remote_kvikio("s3://bucket/table/part-0.parquet", CredentialMode::SharedProcessEnv);
        LocalParquetFile::from_resolved_parquet_source(resolved).unwrap().with_stable_id("part-0")
    }

    #[test]
    fn throughput_planning_widens_remote_chunks_when_the_footer_supplied_row_group_bytes() {
        let file = remote_file()
            .with_row_group_bytes(vec![1024; 7])
            .with_projected_row_group_bytes(vec![512; 7], vec![1024; 7])
            .with_row_group_rows(vec![1; 7]);

        let policy = SourceChunkPlanningPolicy::from_memory_policy_with_mode(
            &remote_source(file),
            one_row_group_policy(),
            SourceChunkPlanningMode::ParquetThroughput,
        );

        assert_eq!(policy.source_chunk_planning_reason(), "parquet_throughput_admitted");
        assert_eq!(policy.max_row_groups_per_chunk, PARQUET_THROUGHPUT_MAX_ROW_GROUPS_PER_CHUNK);
    }

    #[test]
    fn throughput_planning_keeps_one_row_group_chunks_for_remote_files_without_row_group_bytes() {
        let policy = SourceChunkPlanningPolicy::from_memory_policy_with_mode(
            &remote_source(remote_file()),
            one_row_group_policy(),
            SourceChunkPlanningMode::ParquetThroughput,
        );

        assert_eq!(policy.source_chunk_planning_reason(), "parquet_throughput_rejected");
        assert_eq!(policy.max_row_groups_per_chunk, 1);
    }
}
