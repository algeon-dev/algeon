//! Caller-supplied local Parquet facts for dimension-derived pruning.

use super::{DimensionKeyFilter, Expr, KeyRange, LocalParquetSource};

/// Local-filesystem reads a caller supplies for dimension-derived pruning and
/// filtered-scalar-range derivation.
///
/// Mirrors [`super::super::LakehouseFooterPruningResolver`] for the sync,
/// local-path-only case the passes in this module actually need: no async
/// runtime and no object store, because every rewrite decision (candidate
/// discovery, admission, row-group selection) stays in this crate. An
/// implementation answers only "what does this file/row group contain."
///
/// Every method already follows the "no facts ⇒ no rewrite, never a wrong
/// rewrite" rule the pre-resolver code had: a resolver that cannot answer
/// returns `false`/`None`, and every caller in this module already treats
/// that as "skip this candidate" rather than synthesizing an answer.
pub trait LocalParquetPruningResolver {
    /// Whether the currently-selected row groups across `source.files` sum to
    /// at most `max_dimension_rows`. `false` also covers "could not be
    /// determined" (an unreadable footer, or a file that is not local).
    fn dimension_scan_admitted(&self, source: &LocalParquetSource, max_dimension_rows: u64) -> bool;

    /// The dimension key set satisfying `predicate` on `key_column`, decoded
    /// from `source`'s currently-selected row groups and narrowed to the
    /// `projection_indices` file columns.
    ///
    /// `None` means the source could not be read or decoded. `Some(None)` is
    /// a proven-empty key set: every row was rejected by `predicate`.
    fn dimension_key_filter(
        &self,
        source: &LocalParquetSource,
        key_column: &str,
        predicate: &Expr,
        projection_indices: &[usize],
    ) -> Option<Option<DimensionKeyFilter>>;

    /// One inclusive key range per currently-selected row group, per file (in
    /// `source.files` order), for the root file column at `key_column_index`.
    ///
    /// `None` for the whole call means the source could not be read. `None`
    /// for one row group means unknown statistics for that row group
    /// specifically; callers treat an unknown row group as "keep it."
    fn target_row_group_key_ranges(
        &self,
        source: &LocalParquetSource,
        key_column_index: usize,
    ) -> Option<Vec<Vec<Option<KeyRange>>>>;

    /// The inclusive `i64` range of `expr` over `source` rows satisfying
    /// `predicate` (when supplied), decoded from currently-selected row
    /// groups and narrowed to the `projection_indices` file columns.
    fn filtered_i64_expr_range(
        &self,
        source: &LocalParquetSource,
        expr: &Expr,
        predicate: Option<&Expr>,
        projection_indices: &[usize],
    ) -> Option<Option<KeyRange>>;
}

/// A resolver that supplies no facts.
///
/// For callers that do not want dimension-derived pruning or filtered-range
/// derivation to read anything: every method answers "unknown," so every
/// pass in this module skips rather than applying a rewrite built on nothing.
#[cfg(test)]
#[derive(Debug, Clone, Copy, Default)]
pub struct NoDimensionPruningFacts;

#[cfg(test)]
impl LocalParquetPruningResolver for NoDimensionPruningFacts {
    fn dimension_scan_admitted(&self, _source: &LocalParquetSource, _max_dimension_rows: u64) -> bool {
        false
    }

    fn dimension_key_filter(
        &self,
        _source: &LocalParquetSource,
        _key_column: &str,
        _predicate: &Expr,
        _projection_indices: &[usize],
    ) -> Option<Option<DimensionKeyFilter>> {
        None
    }

    fn target_row_group_key_ranges(
        &self,
        _source: &LocalParquetSource,
        _key_column_index: usize,
    ) -> Option<Vec<Vec<Option<KeyRange>>>> {
        None
    }

    fn filtered_i64_expr_range(
        &self,
        _source: &LocalParquetSource,
        _expr: &Expr,
        _predicate: Option<&Expr>,
        _projection_indices: &[usize],
    ) -> Option<Option<KeyRange>> {
        None
    }
}
