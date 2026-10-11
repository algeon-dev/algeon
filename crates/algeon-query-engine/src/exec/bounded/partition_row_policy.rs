//! Row-count partitioning bounds for bounded joins and source-adjacent aggregates.
//!
//! Byte targets (`partition_target_policy`) size a partition by its input bytes.
//! Some cuDF operations allocate per input row or per key match rather than per
//! byte, and this module supplies the row-based bounds those operations need.
//! Like the byte targets, these values shape work; the query domain's allocator
//! alone enforces the cap.

use query_planning::execution_contracts::MemoryControlPolicy;

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

    /// Rows of the larger input that one partition of a mixed join (equality
    /// keys plus a residual predicate) may hold.
    ///
    /// cuDF's mixed join evaluates the residual AST over every equality match
    /// and its size-data pass allocates per probe row, so a byte target over
    /// the inputs does not bound that temporary. The row target grows with the
    /// cap between a floor that keeps small grants from over-partitioning and a
    /// ceiling at which the per-row temporaries stop being the limiting term.
    pub(in crate::exec::bounded) fn mixed_join_target_rows(self, memory_policy: MemoryControlPolicy) -> u64 {
        self.mixed_join_target_rows_for_budget(memory_policy.device_budget_bytes)
    }

    pub(in crate::exec::bounded) fn mixed_join_target_rows_for_budget(self, device_budget_bytes: u64) -> u64 {
        (device_budget_bytes / self.mixed_join_target_rows_budget_byte_divisor)
            .clamp(self.mixed_join_min_target_rows, self.mixed_join_max_target_rows)
    }

    /// Partition count for a partitioned mixed join: the byte-derived count,
    /// raised when the larger input exceeds `target_rows` per partition. The
    /// result is at least two (one partition would copy the inputs for no
    /// split) and at most the logical partition ceiling that the partitioned
    /// join driver also applies to retry repartitioning.
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

    /// Floor for a build side partitioned while its input streams in. Two is
    /// the smallest split under which a probe chunk pairs with less than the
    /// whole build; one partition would only add a copy.
    pub(in crate::exec::bounded) fn hash_frame_stream_min_partitions(self) -> usize {
        self.hash_frame_stream_min_partitions
    }

    /// Floor for the group-key partition count of a source-adjacent aggregate.
    /// It follows the chunk count so a many-chunk input spreads its retained
    /// partials over more partitions than small per-chunk byte estimates would
    /// pick, and it is capped so partition bookkeeping does not grow with
    /// every chunk.
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
}
