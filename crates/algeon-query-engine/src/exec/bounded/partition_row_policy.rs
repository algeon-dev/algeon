use crate::runtime::MemoryControlPolicy;

/// Internal row-count policy for bounded partitioned work.
///
/// This policy covers cardinality-driven partitioning decisions for joins and
/// source-adjacent aggregates. Keep it separate from byte-target sizing so
/// row-count heuristics do not look like user-facing memory controls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::exec::bounded) struct BoundedPartitionRowPolicy {
    mixed_join_min_target_rows: u64,
    mixed_join_max_target_rows: u64,
    mixed_join_target_rows_budget_byte_divisor: u64,
    mixed_join_max_partitions: usize,
    hash_frame_stream_min_partitions: usize,
    source_adjacent_partitioned_aggregate_min_partitions: usize,
    source_adjacent_partitioned_aggregate_max_min_partitions: usize,
}

impl BoundedPartitionRowPolicy {
    pub(in crate::exec::bounded) const DEFAULT_MIXED_JOIN_MIN_TARGET_ROWS: u64 = 50_000;
    pub(in crate::exec::bounded) const DEFAULT_MIXED_JOIN_MAX_TARGET_ROWS: u64 = 16_000_000;
    pub(in crate::exec::bounded) const DEFAULT_MIXED_JOIN_TARGET_ROWS_BUDGET_BYTE_DIVISOR: u64 = 8 * 1024;
    pub(in crate::exec::bounded) const DEFAULT_MIXED_JOIN_MAX_PARTITIONS: usize = 1024;
    pub(in crate::exec::bounded) const DEFAULT_HASH_FRAME_STREAM_MIN_PARTITIONS: usize = 2;
    pub(in crate::exec::bounded) const DEFAULT_SOURCE_ADJACENT_PARTITIONED_AGGREGATE_MIN_PARTITIONS: usize = 1;
    pub(in crate::exec::bounded) const DEFAULT_SOURCE_ADJACENT_PARTITIONED_AGGREGATE_MAX_MIN_PARTITIONS: usize = 32;

    pub(in crate::exec::bounded) fn mixed_join_target_rows(self, memory_policy: MemoryControlPolicy) -> u64 {
        self.mixed_join_target_rows_for_budget(memory_policy.device_budget_bytes)
    }

    pub(in crate::exec::bounded) fn mixed_join_target_rows_for_budget(self, device_budget_bytes: u64) -> u64 {
        (device_budget_bytes / self.mixed_join_target_rows_budget_byte_divisor)
            .clamp(self.mixed_join_min_target_rows, self.mixed_join_max_target_rows)
    }

    pub(in crate::exec::bounded) fn mixed_join_partition_count(
        self,
        byte_partitions: usize,
        target_rows: u64,
        left_rows: u64,
        right_rows: u64,
    ) -> usize {
        let row_basis = left_rows.max(right_rows).max(1);
        let row_partitions = row_basis.saturating_add(target_rows.saturating_sub(1)) / target_rows.max(1);
        let row_partitions = row_partitions.clamp(2, self.mixed_join_max_partitions as u64) as usize;
        byte_partitions.max(row_partitions)
    }

    pub(in crate::exec::bounded) fn hash_frame_stream_min_partitions(self) -> usize {
        self.hash_frame_stream_min_partitions
    }

    pub(in crate::exec::bounded) fn source_adjacent_partitioned_aggregate_min_partition_count(
        self,
        chunk_count: usize,
    ) -> usize {
        chunk_count.clamp(
            self.source_adjacent_partitioned_aggregate_min_partitions,
            self.source_adjacent_partitioned_aggregate_max_min_partitions,
        )
    }
}

impl Default for BoundedPartitionRowPolicy {
    fn default() -> Self {
        Self {
            mixed_join_min_target_rows: Self::DEFAULT_MIXED_JOIN_MIN_TARGET_ROWS,
            mixed_join_max_target_rows: Self::DEFAULT_MIXED_JOIN_MAX_TARGET_ROWS,
            mixed_join_target_rows_budget_byte_divisor: Self::DEFAULT_MIXED_JOIN_TARGET_ROWS_BUDGET_BYTE_DIVISOR,
            mixed_join_max_partitions: Self::DEFAULT_MIXED_JOIN_MAX_PARTITIONS,
            hash_frame_stream_min_partitions: Self::DEFAULT_HASH_FRAME_STREAM_MIN_PARTITIONS,
            source_adjacent_partitioned_aggregate_min_partitions:
                Self::DEFAULT_SOURCE_ADJACENT_PARTITIONED_AGGREGATE_MIN_PARTITIONS,
            source_adjacent_partitioned_aggregate_max_min_partitions:
                Self::DEFAULT_SOURCE_ADJACENT_PARTITIONED_AGGREGATE_MAX_MIN_PARTITIONS,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixed_join_row_target_stays_conservative_for_small_budgets() {
        assert_eq!(
            BoundedPartitionRowPolicy::default().mixed_join_target_rows_for_budget(256 * 1024 * 1024),
            BoundedPartitionRowPolicy::DEFAULT_MIXED_JOIN_MIN_TARGET_ROWS
        );
    }

    #[test]
    fn mixed_join_row_target_scales_with_large_device_budgets() {
        assert_eq!(
            BoundedPartitionRowPolicy::default().mixed_join_target_rows_for_budget(95 * 1024 * 1024 * 1024),
            12_451_840
        );
        assert_eq!(
            BoundedPartitionRowPolicy::default().mixed_join_target_rows_for_budget(256 * 1024 * 1024 * 1024),
            BoundedPartitionRowPolicy::DEFAULT_MIXED_JOIN_MAX_TARGET_ROWS
        );
    }

    #[test]
    fn mixed_join_partition_count_preserves_byte_partition_floor_and_row_cap() {
        let policy = BoundedPartitionRowPolicy::default();

        assert_eq!(policy.mixed_join_partition_count(4, 50_000, 1, 1), 4);
        assert_eq!(policy.mixed_join_partition_count(2, 50_000, 120_000, 1), 3);
        assert_eq!(
            policy.mixed_join_partition_count(2, 1, u64::MAX, 1),
            BoundedPartitionRowPolicy::DEFAULT_MIXED_JOIN_MAX_PARTITIONS
        );
    }

    #[test]
    fn hash_frame_stream_min_partitions_preserves_stream_floor() {
        assert_eq!(
            BoundedPartitionRowPolicy::default().hash_frame_stream_min_partitions(),
            BoundedPartitionRowPolicy::DEFAULT_HASH_FRAME_STREAM_MIN_PARTITIONS
        );
    }

    #[test]
    fn source_adjacent_partitioned_aggregate_min_partition_count_tracks_chunk_count_with_bounds() {
        let policy = BoundedPartitionRowPolicy::default();

        assert_eq!(policy.source_adjacent_partitioned_aggregate_min_partition_count(0), 1);
        assert_eq!(policy.source_adjacent_partitioned_aggregate_min_partition_count(7), 7);
        assert_eq!(policy.source_adjacent_partitioned_aggregate_min_partition_count(64), 32);
    }
}
