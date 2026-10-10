use super::QueryMetrics;

impl QueryMetrics {
    /// Records a resident DataFrame memory observation in bytes.
    pub fn record_device_memory_observation(&mut self, bytes: u64) {
        self.peak_device_memory_bytes = Some(self.peak_device_memory_bytes.map_or(bytes, |peak| peak.max(bytes)));
        self.resident_dataframe_peak_bytes =
            Some(self.resident_dataframe_peak_bytes.map_or(bytes, |peak| peak.max(bytes)));
    }

    /// Records the allocation high-water mark of the attempt's allocation
    /// domain in bytes. The mark covers every plan of the attempt, not only
    /// the plan these metrics belong to.
    pub fn record_allocator_high_water_bytes(&mut self, bytes: u64) {
        self.allocator_high_water_bytes = Some(self.allocator_high_water_bytes.map_or(bytes, |peak| peak.max(bytes)));
    }
}
