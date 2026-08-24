/// Internal policy for source-adjacent batching and bounded read retries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::exec::bounded) struct SourceReadBudgetPolicy {
    source_read_budget_correction_slack_bytes: u64,
}

impl SourceReadBudgetPolicy {
    const DEFAULT_SOURCE_READ_BUDGET_CORRECTION_SLACK_BYTES: u64 = 8 * 1024 * 1024;

    pub(in crate::exec::bounded) const fn source_read_budget_correction_slack_bytes(self) -> u64 {
        self.source_read_budget_correction_slack_bytes
    }

    pub(in crate::exec::bounded) fn source_read_budget_correction_lower_bound(
        self,
        previous_budget_bytes: u64,
        peak_aligned_bytes: u64,
        largest_failed_allocation_bytes: u64,
    ) -> u64 {
        previous_budget_bytes.saturating_mul(2).max(
            peak_aligned_bytes
                .saturating_add(largest_failed_allocation_bytes)
                .saturating_add(self.source_read_budget_correction_slack_bytes),
        )
    }
}

impl Default for SourceReadBudgetPolicy {
    fn default() -> Self {
        Self { source_read_budget_correction_slack_bytes: Self::DEFAULT_SOURCE_READ_BUDGET_CORRECTION_SLACK_BYTES }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_read_budget_correction_preserves_doubling_and_slack_floor() {
        let policy = SourceReadBudgetPolicy::default();

        assert_eq!(policy.source_read_budget_correction_lower_bound(16 * 1024 * 1024, 32, 16), 32 * 1024 * 1024);
        assert_eq!(
            policy.source_read_budget_correction_lower_bound(64, 128, 256),
            128 + 256 + SourceReadBudgetPolicy::DEFAULT_SOURCE_READ_BUDGET_CORRECTION_SLACK_BYTES
        );
        assert_eq!(
            policy.source_read_budget_correction_slack_bytes(),
            SourceReadBudgetPolicy::DEFAULT_SOURCE_READ_BUDGET_CORRECTION_SLACK_BYTES
        );
    }
}
