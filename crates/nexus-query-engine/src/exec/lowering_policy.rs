/// Internal policy for lowering native IR expressions into cudf expressions.
///
/// These gates avoid creating execution expressions whose shape is worse for
/// cudf than the original IR operator. Keep them internal until lower-level
/// cudf diagnostics can report authoritative expression strategy choices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::exec) struct ExpressionLoweringPolicy {
    row_predicate_in_list_expansion_limit: usize,
}

impl ExpressionLoweringPolicy {
    pub(in crate::exec) const DEFAULT_ROW_PREDICATE_IN_LIST_EXPANSION_LIMIT: usize = 128;

    pub(in crate::exec) fn filter_in_list_should_use_row_predicate_tree(self, list_len: usize) -> bool {
        list_len <= self.row_predicate_in_list_expansion_limit
    }

    pub(in crate::exec) fn filter_in_list_should_use_specialized_series(self, list_len: usize) -> bool {
        !self.filter_in_list_should_use_row_predicate_tree(list_len)
    }
}

impl Default for ExpressionLoweringPolicy {
    fn default() -> Self {
        Self { row_predicate_in_list_expansion_limit: Self::DEFAULT_ROW_PREDICATE_IN_LIST_EXPANSION_LIMIT }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn in_list_row_predicate_expansion_allows_limit_but_rejects_above_it() {
        let policy = ExpressionLoweringPolicy::default();

        assert!(policy.filter_in_list_should_use_row_predicate_tree(
            ExpressionLoweringPolicy::DEFAULT_ROW_PREDICATE_IN_LIST_EXPANSION_LIMIT
        ));
        assert!(!policy.filter_in_list_should_use_row_predicate_tree(
            ExpressionLoweringPolicy::DEFAULT_ROW_PREDICATE_IN_LIST_EXPANSION_LIMIT + 1
        ));
    }

    #[test]
    fn specialized_in_list_series_starts_above_row_predicate_limit() {
        let policy = ExpressionLoweringPolicy::default();

        assert!(!policy.filter_in_list_should_use_specialized_series(
            ExpressionLoweringPolicy::DEFAULT_ROW_PREDICATE_IN_LIST_EXPANSION_LIMIT
        ));
        assert!(policy.filter_in_list_should_use_specialized_series(
            ExpressionLoweringPolicy::DEFAULT_ROW_PREDICATE_IN_LIST_EXPANSION_LIMIT + 1
        ));
    }
}
