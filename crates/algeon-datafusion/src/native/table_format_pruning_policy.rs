/// Internal limits for lakehouse footer pruning.
///
/// These limits cap metadata-driven pruning work for remote Parquet scans.
/// Keep them internal until planning reports expose enough evidence for stable
/// production tuning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::native) struct TableFormatPruningPolicy {
    max_dimension_rows: usize,
    max_target_files_across_requests: usize,
}

impl TableFormatPruningPolicy {
    pub(in crate::native) const DEFAULT_MAX_DIMENSION_ROWS: usize = 25_000_000;
    pub(in crate::native) const DEFAULT_MAX_TARGET_FILES_ACROSS_REQUESTS: usize = 4_000;

    pub(in crate::native) fn dimension_row_guard_exceeded(self, total_rows: usize) -> bool {
        total_rows > self.max_dimension_rows
    }

    pub(in crate::native) fn max_dimension_rows(self) -> usize {
        self.max_dimension_rows
    }

    pub(in crate::native) fn max_target_files_across_requests(self) -> usize {
        self.max_target_files_across_requests
    }
}

impl Default for TableFormatPruningPolicy {
    fn default() -> Self {
        Self {
            max_dimension_rows: Self::DEFAULT_MAX_DIMENSION_ROWS,
            max_target_files_across_requests: Self::DEFAULT_MAX_TARGET_FILES_ACROSS_REQUESTS,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dimension_row_guard_allows_limit_but_rejects_above_it() {
        let policy = TableFormatPruningPolicy::default();

        assert!(!policy.dimension_row_guard_exceeded(TableFormatPruningPolicy::DEFAULT_MAX_DIMENSION_ROWS));
        assert!(policy.dimension_row_guard_exceeded(TableFormatPruningPolicy::DEFAULT_MAX_DIMENSION_ROWS + 1));
    }
}
