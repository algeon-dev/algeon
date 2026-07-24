use super::QueryMetrics;

impl QueryMetrics {
    /// Records one adaptive retry action at the operator that performs it.
    pub fn record_adaptive_retry(&mut self) {
        self.adaptive_retry_count = self.adaptive_retry_count.saturating_add(1);
    }

    /// Records one adaptive spill action at the operator that performs it.
    pub fn record_adaptive_spill(&mut self) {
        self.adaptive_spill_count = self.adaptive_spill_count.saturating_add(1);
    }

    /// Records one adaptive downgrade action at the operator that performs it.
    pub fn record_adaptive_downgrade(&mut self) {
        self.adaptive_downgrade_count = self.adaptive_downgrade_count.saturating_add(1);
    }

    /// Records one adaptive memory-history lookup.
    pub fn record_adaptive_memory_history_lookup(&mut self, hit: bool) {
        self.adaptive_memory_history_lookup_count = self.adaptive_memory_history_lookup_count.saturating_add(1);
        if hit {
            self.adaptive_memory_history_hit_count = self.adaptive_memory_history_hit_count.saturating_add(1);
        } else {
            self.adaptive_memory_history_miss_count = self.adaptive_memory_history_miss_count.saturating_add(1);
        }
    }
}
