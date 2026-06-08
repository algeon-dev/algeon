use super::memory::SourceTaskMemoryModel;
use crate::{
    runtime::{MemoryControlPolicy, SourceChunkPlanningMode},
    source::{LocalParquetFile, LocalParquetSource, ParquetReadStrategy},
};

/// Row-group cap used by local-throughput source planning when the configured
/// policy is the conservative one-row-group shape.
pub const LOCAL_PARQUET_THROUGHPUT_MAX_ROW_GROUPS_PER_CHUNK: usize = 1024;
/// Minimum bounded device budget before local-throughput planning can raise the
/// row-group cap. Smaller budgets keep conservative source chunks.
pub const LOCAL_PARQUET_THROUGHPUT_MIN_DEVICE_BUDGET_BYTES: u64 = 32 * 1024 * 1024 * 1024;

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
        Self::for_source_task(memory_policy, model).with_source_planning_mode(source, memory_policy, mode)
    }

    /// Derives a source policy from memory policy and source planning mode.
    pub fn from_memory_policy_with_mode(
        source: &LocalParquetSource,
        memory_policy: MemoryControlPolicy,
        mode: SourceChunkPlanningMode,
    ) -> Self {
        Self::from(memory_policy).with_source_planning_mode(source, memory_policy, mode)
    }

    pub(super) fn effective_max_row_groups_per_chunk(self) -> usize {
        self.max_row_groups_per_chunk.max(1)
    }

    fn with_source_planning_mode(
        mut self,
        source: &LocalParquetSource,
        memory_policy: MemoryControlPolicy,
        mode: SourceChunkPlanningMode,
    ) -> Self {
        if mode != SourceChunkPlanningMode::LocalParquetThroughput {
            return self;
        }
        self.source_chunk_planning_mode = mode;
        if !local_parquet_throughput_planning_admitted(source, memory_policy, self) {
            self.source_chunk_planning_reason = "local_parquet_throughput_rejected";
            return self;
        }
        self.max_row_groups_per_chunk =
            self.max_row_groups_per_chunk.max(LOCAL_PARQUET_THROUGHPUT_MAX_ROW_GROUPS_PER_CHUNK);
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
) -> bool {
    let MemoryControlPolicy::Bounded { device_budget_bytes, .. } = memory_policy else {
        return false;
    };
    if device_budget_bytes < LOCAL_PARQUET_THROUGHPUT_MIN_DEVICE_BUDGET_BYTES {
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
                file.resolved_source.read_strategy,
                ParquetReadStrategy::LocalKvikio | ParquetReadStrategy::LocalCacheKvikio
            )
            && local_parquet_throughput_file_has_source_read_bytes(source, file)
    })
}

fn local_parquet_throughput_file_uses_supported_reader_path(file: &LocalParquetFile) -> bool {
    match file.resolved_source.read_strategy {
        ParquetReadStrategy::LocalKvikio => file.location.is_local_path(),
        ParquetReadStrategy::LocalCacheKvikio => file.path.is_file(),
        ParquetReadStrategy::RemoteHostBuffer | ParquetReadStrategy::RemoteDirect => false,
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
