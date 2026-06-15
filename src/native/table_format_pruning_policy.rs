/// Internal policy for table-format pruning and filter-elision gates.
///
/// These limits cap metadata-driven pruning work for remote table-format scans.
/// Keep them internal until planning reports expose enough evidence for stable
/// production tuning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::native) struct TableFormatPruningPolicy {
    filter_elision_min_source_rows: u64,
    max_dimension_rows: usize,
    max_target_files_across_requests: usize,
}

impl TableFormatPruningPolicy {
    pub(in crate::native) const DEFAULT_FILTER_ELISION_MIN_SOURCE_ROWS: u64 = 10_000_000;
    pub(in crate::native) const DEFAULT_MAX_DIMENSION_ROWS: usize = 25_000_000;
    pub(in crate::native) const DEFAULT_MAX_TARGET_FILES_ACROSS_REQUESTS: usize = 4_000;

    pub(in crate::native) fn reader_pushdown_can_elide_native_filter(self, known_source_rows: Option<u64>) -> bool {
        known_source_rows.is_some_and(|rows| rows >= self.filter_elision_min_source_rows)
    }

    pub(in crate::native) fn dimension_row_guard_exceeded(self, total_rows: usize) -> bool {
        total_rows > self.max_dimension_rows
    }

    pub(in crate::native) fn max_dimension_rows(self) -> usize {
        self.max_dimension_rows
    }

    pub(in crate::native) fn target_file_guard_exceeded(self, target_file_count: usize) -> bool {
        target_file_count > self.max_target_files_across_requests
    }

    pub(in crate::native) fn max_target_files_across_requests(self) -> usize {
        self.max_target_files_across_requests
    }
}

impl Default for TableFormatPruningPolicy {
    fn default() -> Self {
        Self {
            filter_elision_min_source_rows: Self::DEFAULT_FILTER_ELISION_MIN_SOURCE_ROWS,
            max_dimension_rows: Self::DEFAULT_MAX_DIMENSION_ROWS,
            max_target_files_across_requests: Self::DEFAULT_MAX_TARGET_FILES_ACROSS_REQUESTS,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reader_pushdown_filter_elision_requires_large_known_source() {
        let policy = TableFormatPruningPolicy::default();

        assert!(!policy.reader_pushdown_can_elide_native_filter(None));
        assert!(!policy.reader_pushdown_can_elide_native_filter(Some(
            TableFormatPruningPolicy::DEFAULT_FILTER_ELISION_MIN_SOURCE_ROWS - 1
        )));
        assert!(policy.reader_pushdown_can_elide_native_filter(Some(
            TableFormatPruningPolicy::DEFAULT_FILTER_ELISION_MIN_SOURCE_ROWS
        )));
    }

    #[test]
    fn dimension_row_guard_allows_limit_but_rejects_above_it() {
        let policy = TableFormatPruningPolicy::default();

        assert!(!policy.dimension_row_guard_exceeded(TableFormatPruningPolicy::DEFAULT_MAX_DIMENSION_ROWS));
        assert!(policy.dimension_row_guard_exceeded(TableFormatPruningPolicy::DEFAULT_MAX_DIMENSION_ROWS + 1));
    }

    #[test]
    fn target_file_guard_allows_limit_but_rejects_above_it() {
        let policy = TableFormatPruningPolicy::default();

        assert!(!policy.target_file_guard_exceeded(TableFormatPruningPolicy::DEFAULT_MAX_TARGET_FILES_ACROSS_REQUESTS));
        assert!(
            policy.target_file_guard_exceeded(TableFormatPruningPolicy::DEFAULT_MAX_TARGET_FILES_ACROSS_REQUESTS + 1)
        );
    }
}
