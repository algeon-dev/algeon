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
}

impl BoundedFinalMergePolicy {
    pub(in crate::exec::bounded) const DEFAULT_PARTITIONED_FINAL_MERGE_MIN_PARTITION_FLOOR: usize = 1;
    pub(in crate::exec::bounded) const DEFAULT_PARTITIONED_FINAL_MERGE_MAX_PARTIAL_PARTITION_FLOOR: usize = 32;
    pub(in crate::exec::bounded) const DEFAULT_FINAL_MERGE_MIN_FAN_IN: usize = 2;
    pub(in crate::exec::bounded) const DEFAULT_FINAL_MERGE_RETRY_FAN_IN_DIVISOR: usize = 2;

    pub(in crate::exec::bounded) fn partitioned_final_merge_partial_partition_floor(
        self,
        partial_count: usize,
    ) -> usize {
        partial_count.clamp(
            self.partitioned_final_merge_min_partition_floor,
            self.partitioned_final_merge_max_partial_partition_floor,
        )
    }

    pub(in crate::exec::bounded) fn reduce_merge_fan_in(self, merge_fan_in: usize) -> Option<usize> {
        if merge_fan_in <= self.final_merge_min_fan_in {
            return None;
        }

        Some(merge_fan_in.div_ceil(self.final_merge_retry_fan_in_divisor).max(self.final_merge_min_fan_in))
    }

    pub(in crate::exec::bounded) fn merge_fan_in_is_at_retry_floor(self, merge_fan_in: usize) -> bool {
        merge_fan_in <= self.final_merge_min_fan_in
    }

    pub(in crate::exec::bounded) fn should_stop_compiled_final_merge_fan_in_selection(
        self,
        fan_in: usize,
        selected_bytes: u64,
        partial_bytes: u64,
        threshold_bytes: u64,
    ) -> bool {
        fan_in >= self.final_merge_min_fan_in && selected_bytes.saturating_add(partial_bytes) > threshold_bytes
    }

    pub(in crate::exec::bounded) fn compiled_final_merge_fan_in(self, fan_in: usize, partial_count: usize) -> usize {
        fan_in.clamp(self.final_merge_min_fan_in, partial_count)
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
        assert!(policy.merge_fan_in_is_at_retry_floor(2));
        assert!(!policy.merge_fan_in_is_at_retry_floor(3));
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
