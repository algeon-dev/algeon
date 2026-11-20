//! Fan-in and partition bounds for merging bounded aggregate partials.
//!
//! Chunk execution leaves one partial per committed chunk (or per partition
//! batch). The final merge regroups those partials into the SQL result, and
//! this policy bounds how many partials one merge level reads at once, how
//! the incremental merge batches raw partials before hash-partitioning them,
//! and when the incremental merge folds a partition's slices.
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
    incremental_max_pending_partials: usize,
}

/// What an incremental merge does with one partition's pending slices
/// (`BoundedFinalMergePolicy::incremental_partition_fold`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::exec::bounded) enum IncrementalPartitionFold {
    /// Fold the pending slices into the partition's merged state now.
    Fold,
    /// Fold now unless the attempt's retained-state ledger accepts the pending
    /// bytes as a hold, in which case the slices wait for a later fold.
    FoldUnlessHeld,
    /// Keep collecting slices.
    Keep,
}

impl BoundedFinalMergePolicy {
    pub(in crate::exec::bounded) const DEFAULT_PARTITIONED_FINAL_MERGE_MIN_PARTITION_FLOOR: usize = 1;
    pub(in crate::exec::bounded) const DEFAULT_PARTITIONED_FINAL_MERGE_MAX_PARTIAL_PARTITION_FLOOR: usize = 32;
    pub(in crate::exec::bounded) const DEFAULT_FINAL_MERGE_MIN_FAN_IN: usize = 2;
    pub(in crate::exec::bounded) const DEFAULT_FINAL_MERGE_RETRY_FAN_IN_DIVISOR: usize = 2;
    pub(in crate::exec::bounded) const DEFAULT_INCREMENTAL_PARTITION_BATCH_DIVISOR: u64 = 4;
    pub(in crate::exec::bounded) const DEFAULT_INCREMENTAL_MIN_FLUSH_DIVISOR: u64 = 8;
    pub(in crate::exec::bounded) const DEFAULT_INCREMENTAL_MAX_PENDING_PARTIALS: usize = 16;

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

    /// Smallest pending byte count that folds into a partition's merged state
    /// when the slices cannot be held; above it such a partition folds once its
    /// pending slices match the state.
    fn incremental_min_flush_bytes(self, partition_target_bytes: u64) -> u64 {
        (partition_target_bytes / self.incremental_min_flush_divisor).max(1)
    }

    /// Whether `pending_count` raw partials, or slices of one partition, reach
    /// the count that forces a concatenation or a fold.
    pub(in crate::exec::bounded) fn incremental_pending_partials_full(self, pending_count: usize) -> bool {
        pending_count >= self.incremental_max_pending_partials
    }

    /// When an incremental merge folds one partition's pending slices, judged
    /// after a slice was added.
    ///
    /// A fold reads the partition's merged state and its pending slices, and
    /// each fold hashes the merged state again. The slices therefore wait until
    /// they fill one aggregate partition target or match the merged state,
    /// whichever is larger, or reach the slice count bound. A partition whose
    /// input stays below one target is then hashed once, at the end of the
    /// stream, and one that outgrows it folds each time its pending bytes
    /// match the merged state, so its merged rows are hashed O(log) times.
    /// Slices kept past the eager threshold, the larger of the merged state and
    /// the minimum flush bytes, are held on the attempt's retained-state
    /// ledger; when the ledger refuses them the partition folds there.
    pub(in crate::exec::bounded) fn incremental_partition_fold(
        self,
        pending_count: usize,
        pending_bytes: u64,
        merged_bytes: u64,
        partition_target_bytes: u64,
    ) -> IncrementalPartitionFold {
        if pending_count == 0 {
            return IncrementalPartitionFold::Keep;
        }
        if self.incremental_pending_partials_full(pending_count)
            || pending_bytes >= merged_bytes.max(partition_target_bytes)
        {
            return IncrementalPartitionFold::Fold;
        }
        if pending_bytes >= merged_bytes.max(self.incremental_min_flush_bytes(partition_target_bytes)) {
            return IncrementalPartitionFold::FoldUnlessHeld;
        }
        IncrementalPartitionFold::Keep
    }

    /// The same rule judged before a slice of `incoming_bytes` joins a
    /// partition that arrives already aligned to the merge's partitions. The
    /// pending slices fold first when the slice count is full or the pending
    /// and incoming bytes would pass the larger of the target and the merged
    /// state, so one fold never reads more than that plus the merged state.
    /// Past one incremental batch of bytes the slices stay only as a hold.
    pub(in crate::exec::bounded) fn incremental_aligned_partition_fold_before_push(
        self,
        pending_count: usize,
        pending_bytes: u64,
        merged_bytes: u64,
        incoming_bytes: u64,
        partition_target_bytes: u64,
    ) -> IncrementalPartitionFold {
        if pending_count == 0 {
            return IncrementalPartitionFold::Keep;
        }
        let bytes = pending_bytes.saturating_add(incoming_bytes);
        if self.incremental_pending_partials_full(pending_count) || bytes > merged_bytes.max(partition_target_bytes) {
            return IncrementalPartitionFold::Fold;
        }
        if bytes > self.incremental_partition_batch_bytes(partition_target_bytes) {
            return IncrementalPartitionFold::FoldUnlessHeld;
        }
        IncrementalPartitionFold::Keep
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
            incremental_max_pending_partials: Self::DEFAULT_INCREMENTAL_MAX_PENDING_PARTIALS,
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
}
