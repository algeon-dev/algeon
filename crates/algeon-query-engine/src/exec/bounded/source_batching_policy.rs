//! Byte and row caps for bounded source reads and source-adjacent aggregate batches.
//!
//! The query domain's allocator is the only enforcer of the attempt's cap (see
//! `website/docs/design/admission.mdx#allocation-enforcement`, "The allocation
//! path"). The policy here decides how much encoded input one read or one raw
//! batch requests at a time, so that the decoded columns, staged copies and
//! partition slices that are live together usually fit under that cap. A miss
//! surfaces as a typed allocation failure that the chunk-split retry handles;
//! nothing in this module reserves memory or vetoes an allocation.

use query_planning::{chunk::SourceChunkPlanningPolicy, execution_contracts::MemoryControlPolicy};

/// Internal batching caps for bounded local-Parquet source work.
///
/// These values describe runtime batching shapes and are not user-facing resource
/// controls. Keep the public/admin surface on bounded memory policy and source
/// planning mode until production evidence shows a stable tuning contract.
///
/// Every method is a pure function of its arguments and the policy constants.
/// Byte inputs are the planner's encoded-size estimates unless a name says
/// `observed` or `decoded`; the working-set models translate between the two.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::exec::bounded) struct BoundedSourceBatchingPolicy {
    source_batch_read_max_planned_bytes: u64,
    source_adjacent_grouped_aggregate_max_operator_row_groups: usize,
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
    pub(in crate::exec::bounded) const DEFAULT_SOURCE_ADJACENT_GROUPED_AGGREGATE_MAX_OPERATOR_ROW_GROUPS: usize = 1024;
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
    /// million input rows is reduced locally and is not hash-partitioned.
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

    /// Encoded-byte target for one generic (not source-adjacent) source batch.
    ///
    /// The base target is the grant-derived `max_source_chunk_bytes` of the
    /// planning policy. `decode_scratch_risk` is set by the caller when the
    /// source carries a reader predicate and some chunk has filter-only column
    /// bytes; the predicate pass then decodes columns that leave no trace in the
    /// output frame, and the fixed planned-read ceiling bounds that pass however
    /// large the grant is. `None` only when the planning policy has no byte
    /// target.
    pub(in crate::exec::bounded) fn generic_source_batch_target_bytes(
        self,
        memory_policy: MemoryControlPolicy,
        decode_scratch_risk: bool,
    ) -> Option<u64> {
        let target = SourceChunkPlanningPolicy::from(memory_policy).max_source_chunk_bytes;
        if decode_scratch_risk {
            target.map(|bytes| bytes.min(self.source_batch_read_max_planned_bytes))
        } else {
            target
        }
    }

    /// Target for a join leaf whose caller supplies its own byte limit. The
    /// result is always finite: an absent limit yields the planned-read ceiling
    /// so a leaf planned without a byte target still batches.
    pub(in crate::exec::bounded) fn capped_source_batch_target_bytes(
        self,
        max_batch_source_read_bytes: Option<u64>,
    ) -> Option<u64> {
        max_batch_source_read_bytes
            .map(|max_source_chunk_bytes| max_source_chunk_bytes.min(self.source_batch_read_max_planned_bytes))
            .or(Some(self.source_batch_read_max_planned_bytes))
    }

    /// Row-group cap for each chunk of a grouped aggregate with a string group
    /// key (applied by `apply_source_adjacent_aggregate_chunk_caps`). The
    /// footer's encoded size of a string column does not bound its decoded
    /// offsets and characters, and libcudf groupby state over string keys grows
    /// with distinct keys, so the chunk is bounded by row-group count as well
    /// as by bytes.
    pub(in crate::exec::bounded) fn source_adjacent_grouped_aggregate_max_operator_row_groups(self) -> usize {
        self.source_adjacent_grouped_aggregate_max_operator_row_groups
    }

    /// Encoded-byte target for one raw source batch, the multi-chunk read that
    /// stays unaggregated until the source-adjacent filter and projections have
    /// run. The policy's `max_source_chunk_bytes` remains an upper bound; the
    /// working-set share below can only lower it.
    pub(in crate::exec::bounded) fn source_adjacent_raw_source_batch_target_bytes(
        self,
        memory_policy: MemoryControlPolicy,
        source_adjacent_grouped_aggregate: bool,
    ) -> Option<u64> {
        let target = self.source_adjacent_raw_source_batch_max_bytes(
            memory_policy.device_budget_bytes,
            source_adjacent_grouped_aggregate,
        );
        Some(memory_policy.max_source_chunk_bytes.min(target))
    }

    /// The raw batch's working-set share divided by the copies that can be
    /// live at once: the decoded input plus the staged copies counted by the
    /// stage copy factor for a grouped aggregate, or the decoded input and one
    /// partial otherwise.
    pub(in crate::exec::bounded) fn source_adjacent_raw_source_batch_max_bytes(
        self,
        device_budget_bytes: u64,
        source_adjacent_grouped_aggregate: bool,
    ) -> u64 {
        let copies =
            if source_adjacent_grouped_aggregate { 1 + self.source_adjacent_raw_batch_stage_copy_factor } else { 2 };
        (self.source_adjacent_raw_batch_working_set_target_bytes(device_budget_bytes) / copies).max(1)
    }

    /// Whether one group-key partition's pending inputs should be concatenated
    /// and partially aggregated now.
    ///
    /// The first bound reached releases the batch. The input count bounds the
    /// `concat_many` fan-in, the row-group count bounds how much source a retry
    /// of that partial has to redo, and the observed device bytes bound the
    /// concatenated frame. Callers report zero observed bytes for a single
    /// pending input, so a lone input waits for company unless its row groups
    /// alone reach the cap.
    pub(in crate::exec::bounded) fn source_adjacent_partition_batch_ready(
        self,
        input_count: usize,
        source_row_group_count: u64,
        observed_device_bytes: u64,
        device_budget_bytes: u64,
    ) -> bool {
        input_count >= self.source_adjacent_partitioned_aggregate_max_batch_fan_in
            || source_row_group_count >= self.source_adjacent_partitioned_aggregate_max_batch_row_groups
            || observed_device_bytes >= self.source_adjacent_partition_batch_byte_limit_for_budget(device_budget_bytes)
    }

    /// Device-byte bound for one partition batch, a fixed share of the cap
    /// clamped to the min and max batch bytes. The same bound applies to all
    /// retained partitions together (`source_adjacent_pending_partitions_ready`),
    /// since their resident partials share one grant.
    pub(in crate::exec::bounded) fn source_adjacent_partition_batch_byte_limit_for_budget(
        self,
        device_budget_bytes: u64,
    ) -> u64 {
        let budget_limit = device_budget_bytes / self.source_adjacent_partitioned_aggregate_batch_budget_divisor;
        budget_limit.clamp(
            self.source_adjacent_partitioned_aggregate_min_batch_bytes,
            self.source_adjacent_partitioned_aggregate_max_batch_bytes,
        )
    }

    /// Byte target for a join leaf read entirely with remote-direct reads and
    /// materialized into one frame (no stages, scan limit or partition
    /// columns). The larger ceiling trades fewer object-store requests against a
    /// larger decoded frame; the caller only selects it when the whole leaf is
    /// concatenated into one frame anyway.
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
    /// can be partially aggregated whole, without hash partitioning first.
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
            source_adjacent_grouped_aggregate_max_operator_row_groups:
                Self::DEFAULT_SOURCE_ADJACENT_GROUPED_AGGREGATE_MAX_OPERATOR_ROW_GROUPS,
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

        assert_eq!(
            policy.generic_source_batch_target_bytes(
                MemoryControlPolicy {
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
    fn generic_source_batches_use_grant_capacity_without_a_default_byte_ceiling() {
        let batching = BoundedSourceBatchingPolicy::default();
        for grant in [256 * 1024 * 1024, 8 * 1024 * 1024 * 1024, 64 * 1024 * 1024 * 1024] {
            let memory = MemoryControlPolicy {
                device_budget_bytes: grant,
                max_source_chunk_bytes: MemoryControlPolicy::DEFAULT_MAX_SOURCE_CHUNK_BYTES,
                max_row_groups_per_chunk: 1,
                max_retry_attempts: 3,
                min_retry_chunk_row_groups: 1,
            };
            assert_eq!(batching.generic_source_batch_target_bytes(memory, false), Some(grant / 4));
            assert_eq!(
                batching.generic_source_batch_target_bytes(memory, true),
                Some((grant / 4).min(4 * 1024 * 1024 * 1024))
            );
        }
    }

    #[test]
    fn source_adjacent_targets_scale_with_grant_and_preserve_explicit_limit() {
        let policy = BoundedSourceBatchingPolicy::default();
        for grant in [256 * 1024 * 1024, 8 * 1024 * 1024 * 1024, 64 * 1024 * 1024 * 1024] {
            assert_eq!(policy.source_adjacent_raw_source_batch_max_bytes(grant, true), grant / 8);
            assert_eq!(policy.source_adjacent_raw_source_batch_max_bytes(grant, false), grant / 4);
            let explicit = MemoryControlPolicy {
                device_budget_bytes: grant,
                max_source_chunk_bytes: 16 * 1024 * 1024,
                max_row_groups_per_chunk: 1,
                max_retry_attempts: 3,
                min_retry_chunk_row_groups: 1,
            };
            assert_eq!(policy.source_adjacent_raw_source_batch_target_bytes(explicit, true), Some(16 * 1024 * 1024));
        }
    }

    #[test]
    fn source_adjacent_partition_batch_byte_limit_tracks_device_budget_with_bounds() {
        let policy = BoundedSourceBatchingPolicy::default();

        assert_eq!(
            policy.source_adjacent_partition_batch_byte_limit_for_budget(16 * 1024 * 1024 * 1024),
            BoundedSourceBatchingPolicy::DEFAULT_SOURCE_ADJACENT_PARTITIONED_AGGREGATE_MAX_BATCH_BYTES
        );
        assert_eq!(
            policy.source_adjacent_partition_batch_byte_limit_for_budget(4 * 1024 * 1024 * 1024),
            512 * 1024 * 1024
        );
        assert_eq!(
            policy.source_adjacent_partition_batch_byte_limit_for_budget(256 * 1024 * 1024),
            BoundedSourceBatchingPolicy::DEFAULT_SOURCE_ADJACENT_PARTITIONED_AGGREGATE_MIN_BATCH_BYTES
        );
    }
}
