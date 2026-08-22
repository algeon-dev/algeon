use super::memory::SourceTaskMemoryModel;
use crate::{
    error::{QueryError, QueryErrorCode, Result as QueryResult},
    execution_contracts::{MemoryControlPolicy, SourceChunkPlanningMode},
    source::{LocalParquetFile, LocalParquetSource, ParquetReadStrategy},
};

/// Row-group cap used by local-throughput source planning when the configured
/// policy is the conservative one-row-group shape.
pub const LOCAL_PARQUET_THROUGHPUT_MAX_ROW_GROUPS_PER_CHUNK: usize =
    SourceChunkPlanningOptions::DEFAULT_LOCAL_PARQUET_THROUGHPUT_MAX_ROW_GROUPS_PER_CHUNK;
/// Minimum bounded device budget before local-throughput planning can raise the
/// row-group cap. Smaller budgets keep conservative source chunks.
pub const LOCAL_PARQUET_THROUGHPUT_MIN_DEVICE_BUDGET_BYTES: u64 =
    SourceChunkPlanningOptions::DEFAULT_LOCAL_PARQUET_THROUGHPUT_MIN_DEVICE_BUDGET_BYTES;

/// Expert policy for local Parquet source chunk planning heuristics.
///
/// This controls when [`SourceChunkPlanningMode::LocalParquetThroughput`] may
/// widen conservative one-row-group chunks. It is intentionally a Rust-only
/// policy until production evidence shows a stable tuning model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct SourceChunkPlanningOptions {
    local_parquet_throughput_max_row_groups_per_chunk: usize,
    local_parquet_throughput_min_device_budget_bytes: u64,
}

impl SourceChunkPlanningOptions {
    pub const DEFAULT_LOCAL_PARQUET_THROUGHPUT_MAX_ROW_GROUPS_PER_CHUNK: usize = 1024;
    pub const DEFAULT_LOCAL_PARQUET_THROUGHPUT_MIN_DEVICE_BUDGET_BYTES: u64 = 32 * 1024 * 1024 * 1024;

    pub fn local_parquet_throughput_max_row_groups_per_chunk(&self) -> usize {
        self.local_parquet_throughput_max_row_groups_per_chunk
    }

    pub fn local_parquet_throughput_min_device_budget_bytes(&self) -> u64 {
        self.local_parquet_throughput_min_device_budget_bytes
    }

    pub fn with_local_parquet_throughput_max_row_groups_per_chunk(mut self, row_groups: usize) -> Self {
        self.local_parquet_throughput_max_row_groups_per_chunk = row_groups;
        self
    }

    pub fn with_local_parquet_throughput_min_device_budget_bytes(mut self, bytes: u64) -> Self {
        self.local_parquet_throughput_min_device_budget_bytes = bytes;
        self
    }

    pub fn validate(&self) -> QueryResult<()> {
        validate_source_chunk_planning_nonzero_usize(
            "local_parquet_throughput_max_row_groups_per_chunk",
            self.local_parquet_throughput_max_row_groups_per_chunk,
        )?;
        validate_source_chunk_planning_nonzero_u64(
            "local_parquet_throughput_min_device_budget_bytes",
            self.local_parquet_throughput_min_device_budget_bytes,
        )?;
        Ok(())
    }
}

impl Default for SourceChunkPlanningOptions {
    fn default() -> Self {
        Self {
            local_parquet_throughput_max_row_groups_per_chunk:
                Self::DEFAULT_LOCAL_PARQUET_THROUGHPUT_MAX_ROW_GROUPS_PER_CHUNK,
            local_parquet_throughput_min_device_budget_bytes:
                Self::DEFAULT_LOCAL_PARQUET_THROUGHPUT_MIN_DEVICE_BUDGET_BYTES,
        }
    }
}

fn validate_source_chunk_planning_nonzero_usize(field: &'static str, value: usize) -> QueryResult<()> {
    if value == 0 {
        return Err(source_chunk_planning_zero_error(field, value));
    }
    Ok(())
}

fn validate_source_chunk_planning_nonzero_u64(field: &'static str, value: u64) -> QueryResult<()> {
    if value == 0 {
        return Err(source_chunk_planning_zero_error(field, value));
    }
    Ok(())
}

fn source_chunk_planning_zero_error(field: &'static str, value: impl std::fmt::Display) -> QueryError {
    QueryError::new(
        QueryErrorCode::InvalidPlan,
        "nexus_query_engine.source_chunk_planning_options.validate",
        "source chunk planning option must be greater than zero",
    )
    .with_fact("field", field)
    .with_fact("value", value)
    .with_fact("reason", "zero")
}

/// Policy that controls how a local Parquet source is split into chunks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourceChunkPlanningPolicy {
    /// Target upper bound, in bytes, for source-read planning per chunk.
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

    pub fn source_chunk_planning_mode(self) -> SourceChunkPlanningMode {
        self.source_chunk_planning_mode
    }

    pub fn source_chunk_planning_reason(self) -> &'static str {
        self.source_chunk_planning_reason
    }

    pub fn mark_downstream_cap(&mut self, reason: &'static str) {
        self.source_chunk_planning_reason = reason;
    }

    /// One-row-group chunking for scan-limited sources with Iceberg position
    /// deletes, where execution must apply deletes before deciding whether the
    /// source-level limit has produced enough visible rows.
    pub fn for_position_delete_scan_limit() -> Self {
        let mut policy = Self::new(None, 1);
        policy.source_chunk_planning_reason = "position_delete_scan_limit";
        policy
    }

    /// Derives a source-task policy from the bounded memory policy and slot model.
    ///
    /// For unbounded execution this returns the default unbounded policy. For
    /// bounded execution the target is the smaller of the configured source
    /// chunk size and `device_budget_bytes / model.live_slot_count()`.
    pub fn for_source_task(memory_policy: MemoryControlPolicy, model: SourceTaskMemoryModel) -> Self {
        let base = Self::from(memory_policy);
        let MemoryControlPolicy::Bounded {
            device_budget_bytes, max_source_chunk_bytes, max_row_groups_per_chunk, ..
        } = memory_policy
        else {
            return base;
        };
        if max_source_chunk_bytes > device_budget_bytes {
            return base;
        }
        let budget_target = device_budget_bytes / model.live_slot_count().max(1);
        let task_target = max_source_chunk_bytes.min(budget_target.max(1));
        let mut policy = Self::new(Some(task_target), max_row_groups_per_chunk);
        policy.source_chunk_planning_reason =
            if task_target < max_source_chunk_bytes { "memory_cap" } else { "configured_policy" };
        policy
    }

    /// Derives a source-task policy and then applies the source chunk planning
    /// mode for a concrete source.
    pub fn for_source_task_with_mode(
        source: &LocalParquetSource,
        memory_policy: MemoryControlPolicy,
        model: SourceTaskMemoryModel,
        mode: SourceChunkPlanningMode,
    ) -> Self {
        Self::for_source_task_with_mode_and_options(
            source,
            memory_policy,
            model,
            mode,
            SourceChunkPlanningOptions::default(),
        )
    }

    /// Derives a source-task policy with explicit local-throughput planning options.
    pub fn for_source_task_with_mode_and_options(
        source: &LocalParquetSource,
        memory_policy: MemoryControlPolicy,
        model: SourceTaskMemoryModel,
        mode: SourceChunkPlanningMode,
        options: SourceChunkPlanningOptions,
    ) -> Self {
        Self::for_source_task(memory_policy, model).with_source_planning_mode(source, memory_policy, mode, options)
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

    /// Derives a source policy with explicit local-throughput planning options.
    pub fn from_memory_policy_with_mode_and_options(
        source: &LocalParquetSource,
        memory_policy: MemoryControlPolicy,
        mode: SourceChunkPlanningMode,
        options: SourceChunkPlanningOptions,
    ) -> Self {
        Self::from(memory_policy).with_source_planning_mode(source, memory_policy, mode, options)
    }

    pub(super) fn effective_max_row_groups_per_chunk(self) -> usize {
        self.max_row_groups_per_chunk.max(1)
    }

    fn with_source_planning_mode(
        mut self,
        source: &LocalParquetSource,
        memory_policy: MemoryControlPolicy,
        mode: SourceChunkPlanningMode,
        options: SourceChunkPlanningOptions,
    ) -> Self {
        if mode != SourceChunkPlanningMode::LocalParquetThroughput {
            return self;
        }
        self.source_chunk_planning_mode = mode;
        if !local_parquet_throughput_planning_admitted(source, memory_policy, self, options) {
            self.source_chunk_planning_reason = "local_parquet_throughput_rejected";
            return self;
        }
        self.max_row_groups_per_chunk =
            self.max_row_groups_per_chunk.max(options.local_parquet_throughput_max_row_groups_per_chunk());
        self.source_chunk_planning_reason = "local_parquet_throughput_admitted";
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
    fn from(policy: MemoryControlPolicy) -> Self {
        match policy {
            MemoryControlPolicy::Unbounded => Self::default(),
            MemoryControlPolicy::Bounded { max_source_chunk_bytes, max_row_groups_per_chunk, .. } => {
                let mut policy = Self::new(Some(max_source_chunk_bytes), max_row_groups_per_chunk);
                policy.source_chunk_planning_reason = "configured_policy";
                policy
            },
        }
    }
}

fn local_parquet_throughput_planning_admitted(
    source: &LocalParquetSource,
    memory_policy: MemoryControlPolicy,
    policy: SourceChunkPlanningPolicy,
    options: SourceChunkPlanningOptions,
) -> bool {
    let MemoryControlPolicy::Bounded { device_budget_bytes, .. } = memory_policy else {
        return false;
    };
    if device_budget_bytes < options.local_parquet_throughput_min_device_budget_bytes() {
        return false;
    }
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
        local_parquet_throughput_file_uses_supported_reader_path(file)
            && file.partition_values.is_empty()
            && matches!(
                file.resolved_source.read_strategy(),
                ParquetReadStrategy::LocalKvikio | ParquetReadStrategy::LocalCacheKvikio
            )
            && local_parquet_throughput_file_has_source_read_bytes(source, file)
    })
}

fn local_parquet_throughput_file_uses_supported_reader_path(file: &LocalParquetFile) -> bool {
    match file.resolved_source.read_strategy() {
        ParquetReadStrategy::LocalKvikio => file.location.is_local_path(),
        ParquetReadStrategy::LocalCacheKvikio => file.path.is_file(),
        ParquetReadStrategy::RemoteDirect => false,
    }
}

fn local_parquet_throughput_file_has_source_read_bytes(source: &LocalParquetSource, file: &LocalParquetFile) -> bool {
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
