//! Fan-in and partition bounds for merging bounded aggregate partials.
//!
//! Chunk execution leaves one partial per committed chunk (or per partition
//! batch). The final merge regroups those partials into the SQL result, and
//! this policy bounds how many partials one merge level reads at once and how
//! the incremental merge batches raw partials before hash-partitioning them.
//! The fan-in floor of two is structural: a level that merges a single partial
//! produces the same partial and makes no progress.

/// Internal bounded final-merge policy.
///
/// These values control how aggregate partials are regrouped after bounded
/// chunk execution. They are derived from bounded memory policy and retry
/// evidence, so keep them internal until final-merge reports expose a stable
/// production tuning contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::exec::bounded) struct BoundedFinalMergePolicy {
    partitioned_final_merge_min_partition_floor: usize,
    partitioned_final_merge_max_partial_partition_floor: usize,
    final_merge_min_fan_in: usize,
    final_merge_retry_fan_in_divisor: usize,
    incremental_partition_batch_divisor: u64,
    incremental_min_flush_divisor: u64,
}

impl BoundedFinalMergePolicy {
    pub(in crate::exec::bounded) const DEFAULT_PARTITIONED_FINAL_MERGE_MIN_PARTITION_FLOOR: usize = 1;
    pub(in crate::exec::bounded) const DEFAULT_PARTITIONED_FINAL_MERGE_MAX_PARTIAL_PARTITION_FLOOR: usize = 32;
    pub(in crate::exec::bounded) const DEFAULT_FINAL_MERGE_MIN_FAN_IN: usize = 2;
    pub(in crate::exec::bounded) const DEFAULT_FINAL_MERGE_RETRY_FAN_IN_DIVISOR: usize = 2;
    pub(in crate::exec::bounded) const DEFAULT_INCREMENTAL_PARTITION_BATCH_DIVISOR: u64 = 4;
    pub(in crate::exec::bounded) const DEFAULT_INCREMENTAL_MIN_FLUSH_DIVISOR: u64 = 8;

    /// Floor on the partitioned final merge's partition count: one per partial,
    /// capped. Many small partials would otherwise collapse into the one or two
    /// partitions their summed bytes justify, and each partition is merged
    /// whole.
    pub(in crate::exec::bounded) fn partitioned_final_merge_partial_partition_floor(
        self,
        partial_count: usize,
    ) -> usize {
        partial_count.clamp(
            self.partitioned_final_merge_min_partition_floor,
            self.partitioned_final_merge_max_partial_partition_floor,
        )
    }

    /// Fan-in for the next attempt after a merge level failed with a
    /// retryable resource error: halved, rounding up, never below the minimum.
    /// `None` means the fan-in was already minimal and this retry path is
    /// exhausted; the caller reports the original failure.
    pub(in crate::exec::bounded) fn reduce_merge_fan_in(self, merge_fan_in: usize) -> Option<usize> {
        if merge_fan_in <= self.final_merge_min_fan_in {
            return None;
        }

        Some(merge_fan_in.div_ceil(self.final_merge_retry_fan_in_divisor).max(self.final_merge_min_fan_in))
    }

    /// Whether admitting the next partial into a compiled merge level would
    /// push the level's input past `threshold_bytes`, the planned source chunk
    /// target. The first two partials are admitted regardless, since a level
    /// below the minimum fan-in cannot make progress.
    pub(in crate::exec::bounded) fn should_stop_compiled_final_merge_fan_in_selection(
        self,
        fan_in: usize,
        selected_bytes: u64,
        partial_bytes: u64,
        threshold_bytes: u64,
    ) -> bool {
        fan_in >= self.final_merge_min_fan_in && selected_bytes.saturating_add(partial_bytes) > threshold_bytes
    }

    /// Clamps a selected fan-in to the minimum and to the partial count, so a
    /// merge over fewer partials than the minimum still runs as one level.
    pub(in crate::exec::bounded) fn compiled_final_merge_fan_in(self, fan_in: usize, partial_count: usize) -> usize {
        fan_in.clamp(self.final_merge_min_fan_in, partial_count)
    }

    /// Raw partial bytes an incremental merge batches before one hash partition.
    pub(in crate::exec::bounded) fn incremental_partition_batch_bytes(self, partition_target_bytes: u64) -> u64 {
        (partition_target_bytes / self.incremental_partition_batch_divisor).max(1)
    }

    /// Smallest pending byte count that folds into a partition's merged state;
    /// above it a partition folds once its pending slices match the state.
    pub(in crate::exec::bounded) fn incremental_min_flush_bytes(self, partition_target_bytes: u64) -> u64 {
        (partition_target_bytes / self.incremental_min_flush_divisor).max(1)
    }
}

impl Default for BoundedFinalMergePolicy {
    fn default() -> Self {
        Self {
            partitioned_final_merge_min_partition_floor: Self::DEFAULT_PARTITIONED_FINAL_MERGE_MIN_PARTITION_FLOOR,
            partitioned_final_merge_max_partial_partition_floor:
                Self::DEFAULT_PARTITIONED_FINAL_MERGE_MAX_PARTIAL_PARTITION_FLOOR,
            final_merge_min_fan_in: Self::DEFAULT_FINAL_MERGE_MIN_FAN_IN,
            final_merge_retry_fan_in_divisor: Self::DEFAULT_FINAL_MERGE_RETRY_FAN_IN_DIVISOR,
            incremental_partition_batch_divisor: Self::DEFAULT_INCREMENTAL_PARTITION_BATCH_DIVISOR,
            incremental_min_flush_divisor: Self::DEFAULT_INCREMENTAL_MIN_FLUSH_DIVISOR,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partitioned_final_merge_partial_partition_floor_tracks_partial_count_with_bounds() {
        let policy = BoundedFinalMergePolicy::default();

        assert_eq!(policy.partitioned_final_merge_partial_partition_floor(0), 1);
        assert_eq!(policy.partitioned_final_merge_partial_partition_floor(7), 7);
        assert_eq!(policy.partitioned_final_merge_partial_partition_floor(64), 32);
    }

    #[test]
    fn reduce_merge_fan_in_preserves_retry_floor_and_halving_shape() {
        let policy = BoundedFinalMergePolicy::default();

        assert_eq!(policy.reduce_merge_fan_in(1), None);
        assert_eq!(policy.reduce_merge_fan_in(2), None);
        assert_eq!(policy.reduce_merge_fan_in(3), Some(2));
        assert_eq!(policy.reduce_merge_fan_in(4), Some(2));
        assert_eq!(policy.reduce_merge_fan_in(5), Some(3));
    }

    #[test]
    fn compiled_final_merge_selection_keeps_two_partial_floor_before_threshold_stop() {
        let policy = BoundedFinalMergePolicy::default();

        assert!(!policy.should_stop_compiled_final_merge_fan_in_selection(0, 100, 1, 100));
        assert!(!policy.should_stop_compiled_final_merge_fan_in_selection(1, 100, 1, 100));
        assert!(!policy.should_stop_compiled_final_merge_fan_in_selection(2, 99, 1, 100));
        assert!(policy.should_stop_compiled_final_merge_fan_in_selection(2, 100, 1, 100));
    }

    #[test]
    fn compiled_final_merge_fan_in_clamps_to_minimum_and_partial_count() {
        let policy = BoundedFinalMergePolicy::default();

        assert_eq!(policy.compiled_final_merge_fan_in(0, 5), 2);
        assert_eq!(policy.compiled_final_merge_fan_in(4, 5), 4);
        assert_eq!(policy.compiled_final_merge_fan_in(99, 5), 5);
    }
}
