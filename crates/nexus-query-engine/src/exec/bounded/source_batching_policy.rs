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
    remote_direct_materialized_join_source_batch_max_bytes: u64,
}

impl BoundedSourceBatchingPolicy {
    pub(in crate::exec::bounded) const DEFAULT_SOURCE_BATCH_READ_MAX_PLANNED_BYTES: u64 = 4 * 1024 * 1024 * 1024;
    pub(in crate::exec::bounded) const DEFAULT_SOURCE_ADJACENT_AGGREGATE_TARGET_CHUNK_BYTES: u64 =
        2 * 1024 * 1024 * 1024;
    pub(in crate::exec::bounded) const DEFAULT_SOURCE_ADJACENT_GROUPED_AGGREGATE_TARGET_CHUNK_BYTES: u64 =
        384 * 1024 * 1024;
    pub(in crate::exec::bounded) const DEFAULT_SOURCE_ADJACENT_GROUPED_AGGREGATE_MAX_OPERATOR_ROW_GROUPS: usize = 1024;
    pub(in crate::exec::bounded) const DEFAULT_SOURCE_ADJACENT_RAW_SOURCE_BATCH_MAX_BYTES: u64 = 64 * 1024 * 1024;
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
        if remote_direct_required {
            if source_adjacent_grouped_aggregate {
                return self.source_adjacent_grouped_aggregate_target_chunk_bytes;
            }
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
            BoundedSourceBatchingPolicy::DEFAULT_SOURCE_ADJACENT_RAW_SOURCE_BATCH_MAX_BYTES
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
