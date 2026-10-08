//! Size guard for the outer-join null-rejection rewrite.

/// Internal policy for outer-join null-rejection rewrites.
///
/// The large-input guard keeps null-introducing right inputs in their original
/// left-join shape unless downstream aggregate structure makes the rewrite
/// useful without materializing large null-extended rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::optimizer) struct OuterJoinNullRejectionPolicy {
    large_null_rejected_right_input_bytes: u64,
}

impl OuterJoinNullRejectionPolicy {
    pub(in crate::optimizer) const DEFAULT_LARGE_NULL_REJECTED_RIGHT_INPUT_BYTES: u64 = 64 * 1024 * 1024;

    /// An unknown size is not large: a left join whose right input has no byte
    /// estimate stays eligible for the rewrite.
    pub(in crate::optimizer) fn null_introducing_input_is_large(self, right_input_bytes: Option<u64>) -> bool {
        right_input_bytes.is_some_and(|bytes| bytes > self.large_null_rejected_right_input_bytes)
    }
}

impl Default for OuterJoinNullRejectionPolicy {
    fn default() -> Self {
        Self { large_null_rejected_right_input_bytes: Self::DEFAULT_LARGE_NULL_REJECTED_RIGHT_INPUT_BYTES }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_introducing_input_guard_rejects_only_above_limit() {
        let policy = OuterJoinNullRejectionPolicy::default();

        assert!(!policy.null_introducing_input_is_large(None));
        assert!(!policy.null_introducing_input_is_large(Some(
            OuterJoinNullRejectionPolicy::DEFAULT_LARGE_NULL_REJECTED_RIGHT_INPUT_BYTES
        )));
        assert!(policy.null_introducing_input_is_large(Some(
            OuterJoinNullRejectionPolicy::DEFAULT_LARGE_NULL_REJECTED_RIGHT_INPUT_BYTES + 1
        )));
    }
}
