use crate::runtime::MemoryControlPolicy;

/// Internal batching caps for bounded local-Parquet source work.
///
/// These values describe runtime batching shapes, not user-facing resource
/// controls. Keep the public/admin surface on bounded memory policy and source
/// planning mode until production evidence shows a stable tuning contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::exec::bounded) struct BoundedSourceBatchingPolicy {
    source_batch_read_max_planned_bytes: u64,
    source_adjacent_aggregate_target_chunk_bytes: u64,
    source_adjacent_grouped_aggregate_target_chunk_bytes: u64,
    source_adjacent_grouped_aggregate_max_operator_row_groups: usize,
    source_adjacent_raw_source_batch_max_bytes: u64,
    remote_direct_source_adjacent_raw_source_batch_max_bytes: u64,
    source_adjacent_partitioned_aggregate_max_batch_fan_in: usize,
    source_adjacent_partitioned_aggregate_max_batch_row_groups: u64,
    source_adjacent_partitioned_aggregate_max_batch_bytes: u64,
    source_adjacent_partitioned_aggregate_min_batch_bytes: u64,
    source_adjacent_partitioned_aggregate_batch_budget_divisor: u64,
    source_adjacent_raw_batch_working_set_budget_divisor: u64,
    source_adjacent_raw_batch_cold_variable_width_row_bytes: u64,
    source_adjacent_raw_batch_stage_copy_factor: u64,
    source_adjacent_local_reduction_max_output_rows_ppm: u64,
    source_adjacent_local_reduction_max_partial_bytes_divisor: u64,
    source_adjacent_local_reduction_min_partition_count: usize,
    source_adjacent_local_reduction_max_partition_occupancy_divisor: usize,
    remote_direct_materialized_join_source_batch_max_bytes: u64,
}

impl BoundedSourceBatchingPolicy {
    pub(in crate::exec::bounded) const DEFAULT_SOURCE_BATCH_READ_MAX_PLANNED_BYTES: u64 = 4 * 1024 * 1024 * 1024;
    pub(in crate::exec::bounded) const DEFAULT_SOURCE_ADJACENT_AGGREGATE_TARGET_CHUNK_BYTES: u64 =
        2 * 1024 * 1024 * 1024;
    pub(in crate::exec::bounded) const DEFAULT_SOURCE_ADJACENT_GROUPED_AGGREGATE_TARGET_CHUNK_BYTES: u64 =
        384 * 1024 * 1024;
    pub(in crate::exec::bounded) const DEFAULT_SOURCE_ADJACENT_GROUPED_AGGREGATE_MAX_OPERATOR_ROW_GROUPS: usize = 1024;
    pub(in crate::exec::bounded) const DEFAULT_SOURCE_ADJACENT_RAW_SOURCE_BATCH_MAX_BYTES: u64 =
        Self::DEFAULT_SOURCE_ADJACENT_AGGREGATE_TARGET_CHUNK_BYTES;
    pub(in crate::exec::bounded) const DEFAULT_REMOTE_DIRECT_SOURCE_ADJACENT_RAW_SOURCE_BATCH_MAX_BYTES: u64 =
        8 * 1024 * 1024 * 1024;
    pub(in crate::exec::bounded) const DEFAULT_SOURCE_ADJACENT_PARTITIONED_AGGREGATE_MAX_BATCH_FAN_IN: usize = 64;
    pub(in crate::exec::bounded) const DEFAULT_SOURCE_ADJACENT_PARTITIONED_AGGREGATE_MAX_BATCH_ROW_GROUPS: u64 = 64;
    pub(in crate::exec::bounded) const DEFAULT_SOURCE_ADJACENT_PARTITIONED_AGGREGATE_MAX_BATCH_BYTES: u64 =
        1024 * 1024 * 1024;
    pub(in crate::exec::bounded) const DEFAULT_SOURCE_ADJACENT_PARTITIONED_AGGREGATE_MIN_BATCH_BYTES: u64 =
        64 * 1024 * 1024;
    pub(in crate::exec::bounded) const DEFAULT_SOURCE_ADJACENT_PARTITIONED_AGGREGATE_BATCH_BUDGET_DIVISOR: u64 = 8;
    pub(in crate::exec::bounded) const DEFAULT_REMOTE_DIRECT_MATERIALIZED_JOIN_SOURCE_BATCH_MAX_BYTES: u64 =
        9 * 1024 * 1024 * 1024;
    /// One raw source batch's modeled working set (decoded input plus the
    /// stage, partition, and slice copies) may use this fraction of the cap.
    pub(in crate::exec::bounded) const DEFAULT_SOURCE_ADJACENT_RAW_BATCH_WORKING_SET_BUDGET_DIVISOR: u64 = 2;
    /// Before any batch has been observed, a variable-width column is modeled
    /// as this many device bytes per row beside its encoded payload: the
    /// per-row offsets or view entries that Parquet's encoded size omits.
    pub(in crate::exec::bounded) const DEFAULT_SOURCE_ADJACENT_RAW_BATCH_COLD_VARIABLE_WIDTH_ROW_BYTES: u64 = 16;
    /// Copies of the staged (filtered and projected) batch that can be live at
    /// once: the stage output, its hash partition, and the partition slices.
    pub(in crate::exec::bounded) const DEFAULT_SOURCE_ADJACENT_RAW_BATCH_STAGE_COPY_FACTOR: u64 = 3;
    /// A batch whose partial aggregate keeps at most this many output rows per
    /// million input rows is reduced locally instead of hash-partitioned.
    pub(in crate::exec::bounded) const DEFAULT_SOURCE_ADJACENT_LOCAL_REDUCTION_MAX_OUTPUT_ROWS_PPM: u64 = 1_000;
    /// Local reduction stays enabled only while one batch's partial output is
    /// at most the working-set target divided by this value.
    pub(in crate::exec::bounded) const DEFAULT_SOURCE_ADJACENT_LOCAL_REDUCTION_MAX_PARTIAL_BYTES_DIVISOR: u64 = 64;
    /// Partition occupancy is evidence of low key cardinality only with at
    /// least this many partitions.
    pub(in crate::exec::bounded) const DEFAULT_SOURCE_ADJACENT_LOCAL_REDUCTION_MIN_PARTITION_COUNT: usize = 8;
    /// A hash-partitioned batch that filled at most `partition_count / this`
    /// partitions held few distinct keys.
    pub(in crate::exec::bounded) const DEFAULT_SOURCE_ADJACENT_LOCAL_REDUCTION_MAX_PARTITION_OCCUPANCY_DIVISOR: usize =
        4;

    pub(in crate::exec::bounded) fn generic_source_batch_target_bytes(
        self,
        memory_policy: MemoryControlPolicy,
        decode_scratch_risk: bool,
    ) -> Option<u64> {
        match (memory_policy, decode_scratch_risk) {
            (MemoryControlPolicy::Bounded { max_source_chunk_bytes, .. }, true) => {
                Some(max_source_chunk_bytes.min(self.source_batch_read_max_planned_bytes))
            },
            (MemoryControlPolicy::Bounded { max_source_chunk_bytes, .. }, false) => Some(max_source_chunk_bytes),
            (MemoryControlPolicy::Unbounded, true) => Some(self.source_batch_read_max_planned_bytes),
            (MemoryControlPolicy::Unbounded, false) => None,
        }
    }

    pub(in crate::exec::bounded) fn capped_source_batch_target_bytes(
        self,
        max_batch_source_read_bytes: Option<u64>,
    ) -> Option<u64> {
        max_batch_source_read_bytes
            .map(|max_source_chunk_bytes| max_source_chunk_bytes.min(self.source_batch_read_max_planned_bytes))
            .or(Some(self.source_batch_read_max_planned_bytes))
    }

    pub(in crate::exec::bounded) fn source_adjacent_aggregate_target_chunk_bytes(self) -> u64 {
        self.source_adjacent_aggregate_target_chunk_bytes
    }

    pub(in crate::exec::bounded) fn source_adjacent_grouped_aggregate_target_chunk_bytes(self) -> u64 {
        self.source_adjacent_grouped_aggregate_target_chunk_bytes
    }

    pub(in crate::exec::bounded) fn source_adjacent_grouped_aggregate_max_operator_row_groups(self) -> usize {
        self.source_adjacent_grouped_aggregate_max_operator_row_groups
    }

    pub(in crate::exec::bounded) fn source_adjacent_raw_source_batch_target_bytes(
        self,
        memory_policy: MemoryControlPolicy,
        remote_direct_required: bool,
        source_adjacent_grouped_aggregate: bool,
    ) -> Option<u64> {
        let raw_cap =
            self.source_adjacent_raw_source_batch_max_bytes(remote_direct_required, source_adjacent_grouped_aggregate);
        match memory_policy {
            MemoryControlPolicy::Bounded { max_source_chunk_bytes, .. } => Some(max_source_chunk_bytes.min(raw_cap)),
            MemoryControlPolicy::Unbounded => Some(raw_cap),
        }
    }

    pub(in crate::exec::bounded) fn source_adjacent_raw_source_batch_max_bytes(
        self,
        remote_direct_required: bool,
        source_adjacent_grouped_aggregate: bool,
    ) -> u64 {
        if source_adjacent_grouped_aggregate {
            return self.source_adjacent_grouped_aggregate_target_chunk_bytes;
        }
        if remote_direct_required {
            return self.remote_direct_source_adjacent_raw_source_batch_max_bytes;
        }
        self.source_adjacent_raw_source_batch_max_bytes
    }

    pub(in crate::exec::bounded) fn source_adjacent_partition_batch_ready(
        self,
        input_count: usize,
        source_row_group_count: u64,
        reserved_bytes: u64,
        device_budget_bytes: Option<u64>,
    ) -> bool {
        input_count >= self.source_adjacent_partitioned_aggregate_max_batch_fan_in
            || source_row_group_count >= self.source_adjacent_partitioned_aggregate_max_batch_row_groups
            || reserved_bytes >= self.source_adjacent_partition_batch_byte_limit_for_budget(device_budget_bytes)
    }

    pub(in crate::exec::bounded) fn source_adjacent_partition_batch_byte_limit_for_budget(
        self,
        device_budget_bytes: Option<u64>,
    ) -> u64 {
        let budget_limit = device_budget_bytes
            .map(|bytes| bytes / self.source_adjacent_partitioned_aggregate_batch_budget_divisor)
            .unwrap_or(self.source_adjacent_partitioned_aggregate_max_batch_bytes);
        budget_limit.clamp(
            self.source_adjacent_partitioned_aggregate_min_batch_bytes,
            self.source_adjacent_partitioned_aggregate_max_batch_bytes,
        )
    }

    pub(in crate::exec::bounded) fn remote_direct_materialized_join_source_batch_max_bytes(self) -> u64 {
        self.remote_direct_materialized_join_source_batch_max_bytes
    }

    /// Working-set bytes one raw source batch may occupy under `device_budget_bytes`.
    pub(in crate::exec::bounded) fn source_adjacent_raw_batch_working_set_target_bytes(
        self,
        device_budget_bytes: u64,
    ) -> u64 {
        (device_budget_bytes / self.source_adjacent_raw_batch_working_set_budget_divisor).max(1)
    }

    /// Modeled working-set bytes per row before any batch of this source has
    /// been observed.
    ///
    /// Fixed-width read columns decode to their Arrow width regardless of
    /// Parquet encoding; each variable-width column adds its per-row entry
    /// beside the footer's projected uncompressed bytes, which are the only
    /// payload evidence available before a read.
    pub(in crate::exec::bounded) fn source_adjacent_raw_batch_cold_working_set_bytes_per_row(
        self,
        fixed_width_bytes_per_row: u64,
        variable_width_columns: u64,
        projected_uncompressed_bytes_per_row: u64,
    ) -> u64 {
        fixed_width_bytes_per_row
            .saturating_add(
                variable_width_columns.saturating_mul(self.source_adjacent_raw_batch_cold_variable_width_row_bytes),
            )
            .saturating_add(projected_uncompressed_bytes_per_row)
            .max(1)
            .saturating_mul(1 + self.source_adjacent_raw_batch_stage_copy_factor)
    }

    /// Modeled working-set bytes per row from observed decoded and staged bytes.
    pub(in crate::exec::bounded) fn source_adjacent_raw_batch_observed_working_set_bytes_per_row(
        self,
        decoded_bytes_per_row: u64,
        staged_bytes_per_row: u64,
    ) -> u64 {
        decoded_bytes_per_row
            .saturating_add(staged_bytes_per_row.saturating_mul(self.source_adjacent_raw_batch_stage_copy_factor))
            .max(1)
    }

    /// Whether a hash-partitioned batch's occupancy proves few distinct keys.
    pub(in crate::exec::bounded) fn source_adjacent_local_reduction_from_partition_occupancy(
        self,
        non_empty_partitions: usize,
        partition_count: usize,
    ) -> bool {
        partition_count >= self.source_adjacent_local_reduction_min_partition_count
            && non_empty_partitions
                <= partition_count / self.source_adjacent_local_reduction_max_partition_occupancy_divisor
    }

    /// Whether one observed batch reduced strongly enough that the next batch
    /// can be partially aggregated whole instead of hash-partitioned first.
    pub(in crate::exec::bounded) fn source_adjacent_local_reduction_effective(
        self,
        input_rows: u64,
        partial_rows: u64,
        partial_bytes: u64,
        device_budget_bytes: u64,
    ) -> bool {
        let max_partial_bytes = self.source_adjacent_raw_batch_working_set_target_bytes(device_budget_bytes)
            / self.source_adjacent_local_reduction_max_partial_bytes_divisor;
        input_rows > 0
            && u128::from(partial_rows) * 1_000_000
                <= u128::from(input_rows) * u128::from(self.source_adjacent_local_reduction_max_output_rows_ppm)
            && partial_bytes <= max_partial_bytes
    }
}

impl Default for BoundedSourceBatchingPolicy {
    fn default() -> Self {
        Self {
            source_batch_read_max_planned_bytes: Self::DEFAULT_SOURCE_BATCH_READ_MAX_PLANNED_BYTES,
            source_adjacent_aggregate_target_chunk_bytes: Self::DEFAULT_SOURCE_ADJACENT_AGGREGATE_TARGET_CHUNK_BYTES,
            source_adjacent_grouped_aggregate_target_chunk_bytes:
                Self::DEFAULT_SOURCE_ADJACENT_GROUPED_AGGREGATE_TARGET_CHUNK_BYTES,
            source_adjacent_grouped_aggregate_max_operator_row_groups:
                Self::DEFAULT_SOURCE_ADJACENT_GROUPED_AGGREGATE_MAX_OPERATOR_ROW_GROUPS,
            source_adjacent_raw_source_batch_max_bytes: Self::DEFAULT_SOURCE_ADJACENT_RAW_SOURCE_BATCH_MAX_BYTES,
            remote_direct_source_adjacent_raw_source_batch_max_bytes:
                Self::DEFAULT_REMOTE_DIRECT_SOURCE_ADJACENT_RAW_SOURCE_BATCH_MAX_BYTES,
            source_adjacent_partitioned_aggregate_max_batch_fan_in:
                Self::DEFAULT_SOURCE_ADJACENT_PARTITIONED_AGGREGATE_MAX_BATCH_FAN_IN,
            source_adjacent_partitioned_aggregate_max_batch_row_groups:
                Self::DEFAULT_SOURCE_ADJACENT_PARTITIONED_AGGREGATE_MAX_BATCH_ROW_GROUPS,
            source_adjacent_partitioned_aggregate_max_batch_bytes:
                Self::DEFAULT_SOURCE_ADJACENT_PARTITIONED_AGGREGATE_MAX_BATCH_BYTES,
            source_adjacent_partitioned_aggregate_min_batch_bytes:
                Self::DEFAULT_SOURCE_ADJACENT_PARTITIONED_AGGREGATE_MIN_BATCH_BYTES,
            source_adjacent_partitioned_aggregate_batch_budget_divisor:
                Self::DEFAULT_SOURCE_ADJACENT_PARTITIONED_AGGREGATE_BATCH_BUDGET_DIVISOR,
            source_adjacent_raw_batch_working_set_budget_divisor:
                Self::DEFAULT_SOURCE_ADJACENT_RAW_BATCH_WORKING_SET_BUDGET_DIVISOR,
            source_adjacent_raw_batch_cold_variable_width_row_bytes:
                Self::DEFAULT_SOURCE_ADJACENT_RAW_BATCH_COLD_VARIABLE_WIDTH_ROW_BYTES,
            source_adjacent_raw_batch_stage_copy_factor: Self::DEFAULT_SOURCE_ADJACENT_RAW_BATCH_STAGE_COPY_FACTOR,
            source_adjacent_local_reduction_max_output_rows_ppm:
                Self::DEFAULT_SOURCE_ADJACENT_LOCAL_REDUCTION_MAX_OUTPUT_ROWS_PPM,
            source_adjacent_local_reduction_max_partial_bytes_divisor:
                Self::DEFAULT_SOURCE_ADJACENT_LOCAL_REDUCTION_MAX_PARTIAL_BYTES_DIVISOR,
            source_adjacent_local_reduction_min_partition_count:
                Self::DEFAULT_SOURCE_ADJACENT_LOCAL_REDUCTION_MIN_PARTITION_COUNT,
            source_adjacent_local_reduction_max_partition_occupancy_divisor:
                Self::DEFAULT_SOURCE_ADJACENT_LOCAL_REDUCTION_MAX_PARTITION_OCCUPANCY_DIVISOR,
            remote_direct_materialized_join_source_batch_max_bytes:
                Self::DEFAULT_REMOTE_DIRECT_MATERIALIZED_JOIN_SOURCE_BATCH_MAX_BYTES,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generic_source_batch_target_preserves_decode_scratch_cap() {
        let policy = BoundedSourceBatchingPolicy::default();

        assert_eq!(policy.generic_source_batch_target_bytes(MemoryControlPolicy::Unbounded, false), None);
        assert_eq!(
            policy.generic_source_batch_target_bytes(MemoryControlPolicy::Unbounded, true),
            Some(BoundedSourceBatchingPolicy::DEFAULT_SOURCE_BATCH_READ_MAX_PLANNED_BYTES)
        );
        assert_eq!(
            policy.generic_source_batch_target_bytes(
                MemoryControlPolicy::Bounded {
                    device_budget_bytes: 128 * 1024 * 1024 * 1024,
                    max_source_chunk_bytes: 32 * 1024 * 1024 * 1024,
                    max_row_groups_per_chunk: 1,
                    max_retry_attempts: 1,
                    min_retry_chunk_row_groups: 1,
                },
                true,
            ),
            Some(BoundedSourceBatchingPolicy::DEFAULT_SOURCE_BATCH_READ_MAX_PLANNED_BYTES)
        );
    }

    #[test]
    fn source_adjacent_remote_direct_cap_preserves_grouped_override() {
        let policy = BoundedSourceBatchingPolicy::default();

        assert_eq!(
            policy.source_adjacent_raw_source_batch_max_bytes(true, true),
            BoundedSourceBatchingPolicy::DEFAULT_SOURCE_ADJACENT_GROUPED_AGGREGATE_TARGET_CHUNK_BYTES
        );
        assert_eq!(
            policy.source_adjacent_raw_source_batch_max_bytes(true, false),
            BoundedSourceBatchingPolicy::DEFAULT_REMOTE_DIRECT_SOURCE_ADJACENT_RAW_SOURCE_BATCH_MAX_BYTES
        );
        assert_eq!(
            policy.source_adjacent_raw_source_batch_max_bytes(false, true),
            BoundedSourceBatchingPolicy::DEFAULT_SOURCE_ADJACENT_GROUPED_AGGREGATE_TARGET_CHUNK_BYTES
        );
        assert_eq!(
            policy.source_adjacent_raw_source_batch_max_bytes(false, false),
            BoundedSourceBatchingPolicy::DEFAULT_SOURCE_ADJACENT_AGGREGATE_TARGET_CHUNK_BYTES
        );
    }

    #[test]
    fn source_adjacent_partition_batch_byte_limit_tracks_device_budget_with_bounds() {
        let policy = BoundedSourceBatchingPolicy::default();

        assert_eq!(
            policy.source_adjacent_partition_batch_byte_limit_for_budget(Some(16 * 1024 * 1024 * 1024)),
            BoundedSourceBatchingPolicy::DEFAULT_SOURCE_ADJACENT_PARTITIONED_AGGREGATE_MAX_BATCH_BYTES
        );
        assert_eq!(
            policy.source_adjacent_partition_batch_byte_limit_for_budget(Some(4 * 1024 * 1024 * 1024)),
            512 * 1024 * 1024
        );
        assert_eq!(
            policy.source_adjacent_partition_batch_byte_limit_for_budget(Some(256 * 1024 * 1024)),
            BoundedSourceBatchingPolicy::DEFAULT_SOURCE_ADJACENT_PARTITIONED_AGGREGATE_MIN_BATCH_BYTES
        );
        assert_eq!(
            policy.source_adjacent_partition_batch_byte_limit_for_budget(None),
            BoundedSourceBatchingPolicy::DEFAULT_SOURCE_ADJACENT_PARTITIONED_AGGREGATE_MAX_BATCH_BYTES
        );
    }
}
