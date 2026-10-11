//! Shape thresholds for lowering native IR expressions into cuDF expressions.
//!
//! The engine's prepared-expression lowering and its filter operator read one
//! policy here so the two agree on which `IN (...)` lists become a row
//! predicate tree and which become a membership kernel; each list takes one
//! of the two paths, never both.

/// Internal policy for lowering native IR expressions into cudf expressions.
///
/// These gates avoid creating execution expressions whose shape is worse for
/// cudf than the original IR operator. Keep them internal until lower-level
/// cudf diagnostics can report authoritative expression strategy choices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[doc(hidden)]
pub struct ExpressionLoweringPolicy {
    /// Largest `IN` list expanded into an `OR` chain of equality nodes. Each
    /// list element becomes one AST node, so the limit bounds the expression
    /// tree depth cuDF has to compile.
    row_predicate_in_list_expansion_limit: usize,
}

impl ExpressionLoweringPolicy {
    pub const DEFAULT_ROW_PREDICATE_IN_LIST_EXPANSION_LIMIT: usize = 128;

    /// Whether a filter `IN` list of `list_len` values is lowered as an
    /// equality chain inside the row predicate tree.
    pub fn filter_in_list_should_use_row_predicate_tree(self, list_len: usize) -> bool {
        list_len <= self.row_predicate_in_list_expansion_limit
    }

    /// The complement of
    /// [`Self::filter_in_list_should_use_row_predicate_tree`]: the list is
    /// evaluated by a membership kernel over the column instead.
    pub fn filter_in_list_should_use_specialized_series(self, list_len: usize) -> bool {
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
