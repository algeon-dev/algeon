use std::time::Duration;

/// Internal policy for lakehouse cache admission, layout, and sidecar locks.
///
/// These values shape lakehouse cache files and lock retry behavior. They are
/// not exposed as user-facing query knobs; production controls remain the
/// lakehouse cache configuration and cache-fill resource policies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::exec::source) struct LakehouseCachePolicy {
    optimized_batch_min_files: usize,
    optimized_batch_max_source_read_bytes: u64,
    optimized_batch_target_row_groups: usize,
    optimized_batch_min_rows_per_group: usize,
    optimized_batch_max_rows_per_group: usize,
    cache_lock_wait: Duration,
    cache_lock_max_attempts: usize,
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

    pub(in crate::exec::source) fn optimized_batch_row_group_size_rows(self, row_count: usize) -> i32 {
        if row_count == 0 {
            return 0;
        }
        let rows = row_count
            .div_ceil(self.optimized_batch_target_row_groups)
            .clamp(self.optimized_batch_min_rows_per_group, self.optimized_batch_max_rows_per_group);
        i32::try_from(rows).unwrap_or(i32::MAX)
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
    fn optimized_batch_admission_preserves_file_and_byte_gates() {
        let policy = LakehouseCachePolicy::default();

        assert!(!policy.optimized_batch_has_min_files(LakehouseCachePolicy::DEFAULT_OPTIMIZED_BATCH_MIN_FILES - 1));
        assert!(policy.optimized_batch_has_min_files(LakehouseCachePolicy::DEFAULT_OPTIMIZED_BATCH_MIN_FILES));
        assert!(policy.optimized_batch_source_read_bytes_allowed(
            LakehouseCachePolicy::DEFAULT_OPTIMIZED_BATCH_MAX_SOURCE_READ_BYTES
        ));
        assert!(!policy.optimized_batch_source_read_bytes_allowed(
            LakehouseCachePolicy::DEFAULT_OPTIMIZED_BATCH_MAX_SOURCE_READ_BYTES + 1
        ));
    }

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

    #[test]
    fn cache_lock_retry_policy_preserves_wait_and_attempts() {
        let policy = LakehouseCachePolicy::default();

        assert_eq!(policy.cache_lock_wait(), LakehouseCachePolicy::DEFAULT_CACHE_LOCK_WAIT);
        assert_eq!(policy.cache_lock_max_attempts(), LakehouseCachePolicy::DEFAULT_CACHE_LOCK_MAX_ATTEMPTS);
        assert_eq!(
            policy.malformed_cache_lock_stale_age(),
            LakehouseCachePolicy::DEFAULT_MALFORMED_CACHE_LOCK_STALE_AGE
        );
    }
}
