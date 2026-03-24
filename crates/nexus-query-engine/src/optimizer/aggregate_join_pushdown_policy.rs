/// Internal cost policy for aggregate-join pushdown rewrites.
///
/// The policy keeps union pre-aggregation rewrites focused on cases where
/// branch-local aggregation is likely to avoid substantial duplicated join work.
/// It is intentionally not part of public optimizer options until diagnostics
/// can support a stable tuning contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::optimizer) struct AggregateJoinPushdownPolicy {
    union_preagg_min_known_union_bytes: u64,
    union_preagg_max_duplicated_other_bytes: u64,
    union_preagg_min_union_to_duplicated_other_ratio: u64,
}

impl AggregateJoinPushdownPolicy {
    pub(in crate::optimizer) const DEFAULT_UNION_PREAGG_MIN_KNOWN_UNION_BYTES: u64 = 128 * 1024 * 1024;
    pub(in crate::optimizer) const DEFAULT_UNION_PREAGG_MAX_DUPLICATED_OTHER_BYTES: u64 = 64 * 1024 * 1024;
    pub(in crate::optimizer) const DEFAULT_UNION_PREAGG_MIN_UNION_TO_DUPLICATED_OTHER_RATIO: u64 = 4;

    pub(in crate::optimizer) fn join_union_preaggregate_admitted(
        self,
        known_union_bytes: u64,
        other_bytes: u64,
        union_input_count: usize,
    ) -> bool {
        let Some(duplicated_other_bytes) = other_bytes.checked_mul(union_input_count as u64) else {
            return false;
        };
        known_union_bytes >= self.union_preagg_min_known_union_bytes
            && other_bytes <= self.union_preagg_max_duplicated_other_bytes
            && known_union_bytes
                >= duplicated_other_bytes.saturating_mul(self.union_preagg_min_union_to_duplicated_other_ratio)
    }
}

impl Default for AggregateJoinPushdownPolicy {
    fn default() -> Self {
        Self {
            union_preagg_min_known_union_bytes: Self::DEFAULT_UNION_PREAGG_MIN_KNOWN_UNION_BYTES,
            union_preagg_max_duplicated_other_bytes: Self::DEFAULT_UNION_PREAGG_MAX_DUPLICATED_OTHER_BYTES,
            union_preagg_min_union_to_duplicated_other_ratio:
                Self::DEFAULT_UNION_PREAGG_MIN_UNION_TO_DUPLICATED_OTHER_RATIO,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn join_union_preaggregate_admits_boundary_case() {
        let policy = AggregateJoinPushdownPolicy::default();

        assert!(policy.join_union_preaggregate_admitted(
            AggregateJoinPushdownPolicy::DEFAULT_UNION_PREAGG_MIN_KNOWN_UNION_BYTES,
            16 * 1024 * 1024,
            2,
        ));
    }

    #[test]
    fn join_union_preaggregate_rejects_when_union_work_is_small() {
        let policy = AggregateJoinPushdownPolicy::default();

        assert!(!policy.join_union_preaggregate_admitted(
            AggregateJoinPushdownPolicy::DEFAULT_UNION_PREAGG_MIN_KNOWN_UNION_BYTES - 1,
            1,
            2,
        ));
    }

    #[test]
    fn join_union_preaggregate_rejects_large_duplicated_side() {
        let policy = AggregateJoinPushdownPolicy::default();

        assert!(!policy.join_union_preaggregate_admitted(
            u64::MAX,
            AggregateJoinPushdownPolicy::DEFAULT_UNION_PREAGG_MAX_DUPLICATED_OTHER_BYTES + 1,
            2,
        ));
    }

    #[test]
    fn join_union_preaggregate_rejects_weak_union_to_duplicate_ratio() {
        let policy = AggregateJoinPushdownPolicy::default();

        assert!(!policy.join_union_preaggregate_admitted(
            AggregateJoinPushdownPolicy::DEFAULT_UNION_PREAGG_MIN_KNOWN_UNION_BYTES,
            20 * 1024 * 1024,
            2,
        ));
    }

    #[test]
    fn join_union_preaggregate_rejects_duplicated_side_overflow() {
        let policy = AggregateJoinPushdownPolicy::default();

        assert!(!policy.join_union_preaggregate_admitted(u64::MAX, u64::MAX, 2));
    }
}
