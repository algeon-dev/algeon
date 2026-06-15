/// Internal reservation accounting policy for bounded source chunks.
///
/// These values are local accounting safeguards around observed source-read
/// memory and source-adjacent operator work. They are not user-facing resource
/// knobs; public control remains the bounded memory policy and source IO gates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::exec::bounded) struct BoundedSourceReservationPolicy {
    source_adjacent_work_reservation_headroom_bytes: u64,
    source_read_budget_correction_slack_bytes: u64,
}

impl BoundedSourceReservationPolicy {
    pub(in crate::exec::bounded) const DEFAULT_SOURCE_ADJACENT_WORK_RESERVATION_HEADROOM_BYTES: u64 = 1;
    pub(in crate::exec::bounded) const DEFAULT_SOURCE_READ_BUDGET_CORRECTION_SLACK_BYTES: u64 = 8 * 1024 * 1024;

    pub(in crate::exec::bounded) fn source_adjacent_unary_work_request_bytes(
        self,
        input_reserved_bytes: u64,
        remaining_bytes: u64,
    ) -> u64 {
        input_reserved_bytes.min(remaining_bytes.saturating_sub(self.source_adjacent_work_reservation_headroom_bytes))
    }

    pub(in crate::exec::bounded) fn source_read_budget_correction_slack_bytes(self) -> u64 {
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

impl Default for BoundedSourceReservationPolicy {
    fn default() -> Self {
        Self {
            source_adjacent_work_reservation_headroom_bytes:
                Self::DEFAULT_SOURCE_ADJACENT_WORK_RESERVATION_HEADROOM_BYTES,
            source_read_budget_correction_slack_bytes: Self::DEFAULT_SOURCE_READ_BUDGET_CORRECTION_SLACK_BYTES,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_adjacent_work_request_keeps_headroom() {
        let policy = BoundedSourceReservationPolicy::default();

        assert_eq!(policy.source_adjacent_unary_work_request_bytes(100, 0), 0);
        assert_eq!(policy.source_adjacent_unary_work_request_bytes(100, 1), 0);
        assert_eq!(policy.source_adjacent_unary_work_request_bytes(100, 2), 1);
        assert_eq!(policy.source_adjacent_unary_work_request_bytes(100, 256), 100);
    }

    #[test]
    fn source_read_budget_correction_preserves_doubling_and_slack_floor() {
        let policy = BoundedSourceReservationPolicy::default();

        assert_eq!(
            policy.source_read_budget_correction_lower_bound(16 * 1024 * 1024, 32, 16),
            32 * 1024 * 1024,
            "previous budget doubling remains the lower bound when it is larger"
        );
        assert_eq!(
            policy.source_read_budget_correction_lower_bound(64, 128, 256),
            128 + 256 + BoundedSourceReservationPolicy::DEFAULT_SOURCE_READ_BUDGET_CORRECTION_SLACK_BYTES
        );
        assert_eq!(
            policy.source_read_budget_correction_slack_bytes(),
            BoundedSourceReservationPolicy::DEFAULT_SOURCE_READ_BUDGET_CORRECTION_SLACK_BYTES
        );
    }
}
