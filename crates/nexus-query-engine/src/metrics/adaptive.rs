use super::QueryMetrics;

impl QueryMetrics {
    /// Adds adaptive runtime action counters.
    pub fn record_adaptive_runtime_action_counts(&mut self, retry_count: u64, spill_count: u64, downgrade_count: u64) {
        self.adaptive_retry_count = self.adaptive_retry_count.saturating_add(retry_count);
        self.adaptive_spill_count = self.adaptive_spill_count.saturating_add(spill_count);
        self.adaptive_downgrade_count = self.adaptive_downgrade_count.saturating_add(downgrade_count);
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
