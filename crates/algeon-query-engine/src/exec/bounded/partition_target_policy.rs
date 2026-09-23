use crate::runtime::MemoryControlPolicy;

/// Internal byte-target policy for bounded partitioning and streaming.
///
/// These targets control how bounded execution splits large aggregate and join
/// work into GPU-sized pieces. They are intentionally not exposed as
/// user-facing knobs; public resource control remains the bounded memory policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::exec::bounded) struct BoundedPartitionTargetPolicy {
    aggregate_partition_min_target_bytes: u64,
    aggregate_partition_max_target_bytes: u64,
    aggregate_partition_budget_divisor: u64,
    partitioned_join_min_target_bytes: u64,
    partitioned_join_max_target_bytes: u64,
    partitioned_join_budget_divisor: u64,
    reusable_hash_join_stream_min_target_bytes: u64,
    reusable_hash_join_stream_max_target_bytes: u64,
    reusable_hash_join_stream_budget_divisor: u64,
    reusable_hash_join_stream_live_divisor: u64,
}

impl BoundedPartitionTargetPolicy {
    pub(in crate::exec::bounded) const DEFAULT_AGGREGATE_PARTITION_MIN_TARGET_BYTES: u64 = 128 * 1024 * 1024;
    pub(in crate::exec::bounded) const DEFAULT_AGGREGATE_PARTITION_MAX_TARGET_BYTES: u64 = 1024 * 1024 * 1024;
    pub(in crate::exec::bounded) const DEFAULT_AGGREGATE_PARTITION_BUDGET_DIVISOR: u64 = 32;
    pub(in crate::exec::bounded) const DEFAULT_PARTITIONED_JOIN_MIN_TARGET_BYTES: u64 = 128 * 1024 * 1024;
    pub(in crate::exec::bounded) const DEFAULT_PARTITIONED_JOIN_MAX_TARGET_BYTES: u64 = 8 * 1024 * 1024 * 1024;
    pub(in crate::exec::bounded) const DEFAULT_PARTITIONED_JOIN_BUDGET_DIVISOR: u64 = 16;
    pub(in crate::exec::bounded) const DEFAULT_REUSABLE_HASH_JOIN_STREAM_MIN_TARGET_BYTES: u64 = 128 * 1024 * 1024;
    pub(in crate::exec::bounded) const DEFAULT_REUSABLE_HASH_JOIN_STREAM_MAX_TARGET_BYTES: u64 = 512 * 1024 * 1024;
    pub(in crate::exec::bounded) const DEFAULT_REUSABLE_HASH_JOIN_STREAM_BUDGET_DIVISOR: u64 = 8;
    pub(in crate::exec::bounded) const DEFAULT_REUSABLE_HASH_JOIN_STREAM_LIVE_DIVISOR: u64 = 16;

    pub(in crate::exec::bounded) fn aggregate_partition_target_bytes(self, device_budget_bytes: u64) -> u64 {
        (device_budget_bytes / self.aggregate_partition_budget_divisor)
            .clamp(self.aggregate_partition_min_target_bytes, self.aggregate_partition_max_target_bytes)
    }

    pub(in crate::exec::bounded) fn partitioned_join_target_bytes(self, device_budget_bytes: u64) -> u64 {
        (device_budget_bytes / self.partitioned_join_budget_divisor)
            .clamp(self.partitioned_join_min_target_bytes, self.partitioned_join_max_target_bytes)
    }

    pub(in crate::exec::bounded) fn reusable_hash_join_stream_target_output_bytes(
        self,
        memory_policy: MemoryControlPolicy,
        remaining_bytes: u64,
    ) -> u64 {
        let policy_target = (memory_policy.device_budget_bytes / self.reusable_hash_join_stream_budget_divisor)
            .clamp(self.reusable_hash_join_stream_min_target_bytes, self.reusable_hash_join_stream_max_target_bytes);
        let live_target = remaining_bytes.saturating_div(self.reusable_hash_join_stream_live_divisor);
        if live_target == 0 { policy_target } else { policy_target.min(live_target) }
    }
}

impl Default for BoundedPartitionTargetPolicy {
    fn default() -> Self {
        Self {
            aggregate_partition_min_target_bytes: Self::DEFAULT_AGGREGATE_PARTITION_MIN_TARGET_BYTES,
            aggregate_partition_max_target_bytes: Self::DEFAULT_AGGREGATE_PARTITION_MAX_TARGET_BYTES,
            aggregate_partition_budget_divisor: Self::DEFAULT_AGGREGATE_PARTITION_BUDGET_DIVISOR,
            partitioned_join_min_target_bytes: Self::DEFAULT_PARTITIONED_JOIN_MIN_TARGET_BYTES,
            partitioned_join_max_target_bytes: Self::DEFAULT_PARTITIONED_JOIN_MAX_TARGET_BYTES,
            partitioned_join_budget_divisor: Self::DEFAULT_PARTITIONED_JOIN_BUDGET_DIVISOR,
            reusable_hash_join_stream_min_target_bytes: Self::DEFAULT_REUSABLE_HASH_JOIN_STREAM_MIN_TARGET_BYTES,
            reusable_hash_join_stream_max_target_bytes: Self::DEFAULT_REUSABLE_HASH_JOIN_STREAM_MAX_TARGET_BYTES,
            reusable_hash_join_stream_budget_divisor: Self::DEFAULT_REUSABLE_HASH_JOIN_STREAM_BUDGET_DIVISOR,
            reusable_hash_join_stream_live_divisor: Self::DEFAULT_REUSABLE_HASH_JOIN_STREAM_LIVE_DIVISOR,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aggregate_partition_target_bytes_tracks_budget_with_bounds() {
        let policy = BoundedPartitionTargetPolicy::default();

        assert_eq!(
            policy.aggregate_partition_target_bytes(2 * 1024 * 1024 * 1024),
            BoundedPartitionTargetPolicy::DEFAULT_AGGREGATE_PARTITION_MIN_TARGET_BYTES
        );
        assert_eq!(policy.aggregate_partition_target_bytes(16 * 1024 * 1024 * 1024), 512 * 1024 * 1024);
        assert_eq!(
            policy.aggregate_partition_target_bytes(128 * 1024 * 1024 * 1024),
            BoundedPartitionTargetPolicy::DEFAULT_AGGREGATE_PARTITION_MAX_TARGET_BYTES
        );
    }

    #[test]
    fn partitioned_join_target_bytes_tracks_budget_with_bounds() {
        let policy = BoundedPartitionTargetPolicy::default();

        assert_eq!(
            policy.partitioned_join_target_bytes(1024 * 1024 * 1024),
            BoundedPartitionTargetPolicy::DEFAULT_PARTITIONED_JOIN_MIN_TARGET_BYTES
        );
        assert_eq!(policy.partitioned_join_target_bytes(8 * 1024 * 1024 * 1024), 512 * 1024 * 1024);
        assert_eq!(policy.partitioned_join_target_bytes(32 * 1024 * 1024 * 1024), 2 * 1024 * 1024 * 1024);
        assert_eq!(
            policy.partitioned_join_target_bytes(95 * 1024 * 1024 * 1024),
            95 * 1024 * 1024 * 1024 / BoundedPartitionTargetPolicy::DEFAULT_PARTITIONED_JOIN_BUDGET_DIVISOR
        );
        assert_eq!(
            policy.partitioned_join_target_bytes(128 * 1024 * 1024 * 1024),
            BoundedPartitionTargetPolicy::DEFAULT_PARTITIONED_JOIN_MAX_TARGET_BYTES
        );
    }

    #[test]
    fn reusable_hash_join_stream_target_respects_policy_and_live_bytes() {
        let policy = BoundedPartitionTargetPolicy::default();
        let memory_policy = MemoryControlPolicy {
            device_budget_bytes: 8 * 1024 * 1024 * 1024,
            max_source_chunk_bytes: 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        };

        assert_eq!(
            policy.reusable_hash_join_stream_target_output_bytes(memory_policy, 0),
            BoundedPartitionTargetPolicy::DEFAULT_REUSABLE_HASH_JOIN_STREAM_MAX_TARGET_BYTES
        );
        let small_budget_policy = MemoryControlPolicy {
            device_budget_bytes: 512 * 1024 * 1024,
            max_source_chunk_bytes: 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        };
        assert_eq!(
            policy.reusable_hash_join_stream_target_output_bytes(small_budget_policy, 0),
            BoundedPartitionTargetPolicy::DEFAULT_REUSABLE_HASH_JOIN_STREAM_MIN_TARGET_BYTES
        );
        assert_eq!(
            policy.reusable_hash_join_stream_target_output_bytes(small_budget_policy, 15),
            BoundedPartitionTargetPolicy::DEFAULT_REUSABLE_HASH_JOIN_STREAM_MIN_TARGET_BYTES
        );
        assert_eq!(
            policy.reusable_hash_join_stream_target_output_bytes(memory_policy, 1024 * 1024 * 1024),
            64 * 1024 * 1024
        );
    }
}
