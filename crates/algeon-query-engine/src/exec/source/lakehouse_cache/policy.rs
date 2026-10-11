use std::time::Duration;

/// Internal policy for lakehouse cache admission, layout, and sidecar locks.
///
/// These values shape lakehouse cache files and lock retry behavior. They are
/// not exposed as user-facing query settings; production controls remain the
/// lakehouse cache configuration and cache-fill resource policies.
///
/// The optimized batch cache rewrites a batch of cache-served chunks into one Parquet file. This
/// policy decides from three numbers only: the batch's chunk count, its estimated source read
/// bytes, and the row count of the frame being written. It reads no memory, GPU or cache
/// occupancy state, and it takes no part in eviction, which `cache` performs by last use against
/// the configured `cache_max_bytes`. Callers build it with `Default`; a given value always
/// returns the same answer for the same inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::exec::source) struct LakehouseCachePolicy {
    /// Smaller batches are not rewritten; the per-file entries already serve them.
    optimized_batch_min_files: usize,
    /// Inclusive cap on a batch's estimated source read bytes. A larger batch is not rewritten.
    optimized_batch_max_source_read_bytes: u64,
    /// Number of row groups the rewritten file aims for, before the row bounds below apply.
    optimized_batch_target_row_groups: usize,
    optimized_batch_min_rows_per_group: usize,
    optimized_batch_max_rows_per_group: usize,
    /// Sleep between attempts to take another owner's fill lock.
    cache_lock_wait: Duration,
    /// Attempts before a fill gives up on the lock with a `lock_timeout` error. With the default
    /// wait this bounds the wait for another owner at about one minute.
    cache_lock_max_attempts: usize,
    /// A lock file with no readable `pid=` line is treated as abandoned once its modification time
    /// is older than this. It covers the gap between a creator's exclusive create and its pid
    /// write, so it must stay longer than that gap.
    malformed_cache_lock_stale_age: Duration,
}

impl LakehouseCachePolicy {
    pub(in crate::exec::source) const DEFAULT_OPTIMIZED_BATCH_MIN_FILES: usize = 8;
    pub(in crate::exec::source) const DEFAULT_OPTIMIZED_BATCH_MAX_SOURCE_READ_BYTES: u64 = 64 * 1024 * 1024;
    pub(in crate::exec::source) const DEFAULT_OPTIMIZED_BATCH_TARGET_ROW_GROUPS: usize = 32;
    pub(in crate::exec::source) const DEFAULT_OPTIMIZED_BATCH_MIN_ROWS_PER_GROUP: usize = 50_000;
    pub(in crate::exec::source) const DEFAULT_OPTIMIZED_BATCH_MAX_ROWS_PER_GROUP: usize = 1_000_000;
    pub(in crate::exec::source) const DEFAULT_CACHE_LOCK_WAIT: Duration = Duration::from_millis(100);
    pub(in crate::exec::source) const DEFAULT_CACHE_LOCK_MAX_ATTEMPTS: usize = 600;
    pub(in crate::exec::source) const DEFAULT_MALFORMED_CACHE_LOCK_STALE_AGE: Duration = Duration::from_millis(250);

    pub(in crate::exec::source) fn optimized_batch_has_min_files(self, file_count: usize) -> bool {
        file_count >= self.optimized_batch_min_files
    }

    pub(in crate::exec::source) fn optimized_batch_source_read_bytes_allowed(self, source_read_bytes: u64) -> bool {
        source_read_bytes <= self.optimized_batch_max_source_read_bytes
    }

    /// Row-group size for the writer: `row_count` divided by the target group count, rounded up and
    /// clamped to the row bounds. Zero rows return 0, which selects the libcudf writer default.
    pub(in crate::exec::source) fn optimized_batch_row_group_size_rows(self, row_count: usize) -> i32 {
        if row_count == 0 {
            return 0;
        }
        let rows = row_count
            .div_ceil(self.optimized_batch_target_row_groups)
            .clamp(self.optimized_batch_min_rows_per_group, self.optimized_batch_max_rows_per_group);
        rows as i32
    }

    pub(in crate::exec::source) fn cache_lock_wait(self) -> Duration {
        self.cache_lock_wait
    }

    pub(in crate::exec::source) fn cache_lock_max_attempts(self) -> usize {
        self.cache_lock_max_attempts
    }

    pub(in crate::exec::source) fn malformed_cache_lock_stale_age(self) -> Duration {
        self.malformed_cache_lock_stale_age
    }
}

impl Default for LakehouseCachePolicy {
    fn default() -> Self {
        Self {
            optimized_batch_min_files: Self::DEFAULT_OPTIMIZED_BATCH_MIN_FILES,
            optimized_batch_max_source_read_bytes: Self::DEFAULT_OPTIMIZED_BATCH_MAX_SOURCE_READ_BYTES,
            optimized_batch_target_row_groups: Self::DEFAULT_OPTIMIZED_BATCH_TARGET_ROW_GROUPS,
            optimized_batch_min_rows_per_group: Self::DEFAULT_OPTIMIZED_BATCH_MIN_ROWS_PER_GROUP,
            optimized_batch_max_rows_per_group: Self::DEFAULT_OPTIMIZED_BATCH_MAX_ROWS_PER_GROUP,
            cache_lock_wait: Self::DEFAULT_CACHE_LOCK_WAIT,
            cache_lock_max_attempts: Self::DEFAULT_CACHE_LOCK_MAX_ATTEMPTS,
            malformed_cache_lock_stale_age: Self::DEFAULT_MALFORMED_CACHE_LOCK_STALE_AGE,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optimized_batch_row_group_size_tracks_target_with_bounds() {
        let policy = LakehouseCachePolicy::default();

        assert_eq!(policy.optimized_batch_row_group_size_rows(0), 0);
        assert_eq!(
            policy.optimized_batch_row_group_size_rows(1),
            LakehouseCachePolicy::DEFAULT_OPTIMIZED_BATCH_MIN_ROWS_PER_GROUP as i32
        );
        assert_eq!(policy.optimized_batch_row_group_size_rows(3_200_000), 100_000);
        assert_eq!(
            policy.optimized_batch_row_group_size_rows(64_000_000),
            LakehouseCachePolicy::DEFAULT_OPTIMIZED_BATCH_MAX_ROWS_PER_GROUP as i32
        );
    }
}
