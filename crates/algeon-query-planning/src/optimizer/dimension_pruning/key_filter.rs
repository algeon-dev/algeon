use std::collections::BTreeSet;

use super::{Expr, RecordBatch, array_i64_value, eval_bool};

/// Inclusive dimension key range used to prune joined fact row groups.
///
/// Exposed so the DataFusion/object-store adapter can share pruning semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeyRange {
    /// Inclusive minimum key value among rows matching the dimension predicate.
    pub min: i64,
    /// Inclusive maximum key value among rows matching the dimension predicate.
    pub max: i64,
}

/// Maximum coalesced runs before falling back to one bounding range.
///
/// Bounds memory/overlap cost while preserving the single bounding-range case.
pub const DIMENSION_KEY_FILTER_MAX_RUNS: usize = DimensionKeyFilter::MAX_RUNS;

/// Dimension keys satisfying a predicate, stored as disjoint inclusive runs.
///
/// Row groups are retained only when a run overlaps their key stats. Contiguous
/// predicates collapse to one bounding `[min,max]` range.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DimensionKeyFilter {
    /// Sorted by `min`, disjoint, non-adjacent, non-empty.
    runs: Vec<KeyRange>,
}

impl DimensionKeyFilter {
    /// Maximum coalesced runs before falling back to one bounding range.
    pub const MAX_RUNS: usize = 4096;

    /// Builds a filter from ascending, unique keys, coalescing contiguous
    /// integers into runs. Collapses to a single bounding run when the run count
    /// would exceed [`DIMENSION_KEY_FILTER_MAX_RUNS`]. Returns `None` when the
    /// input is empty. The caller must guarantee the input is sorted ascending
    /// and deduplicated (a `BTreeSet` iteration order satisfies this).
    fn from_sorted_unique_keys(sorted_unique_keys: &[i64]) -> Option<Self> {
        let (&first, rest) = sorted_unique_keys.split_first()?;
        let mut runs = Vec::new();
        let mut run = KeyRange { min: first, max: first };
        for &value in rest {
            if value == run.max.saturating_add(1) {
                run.max = value;
            } else {
                runs.push(run);
                run = KeyRange { min: value, max: value };
            }
        }
        runs.push(run);
        if runs.len() > Self::MAX_RUNS {
            return Some(Self::from_bounds(KeyRange {
                min: runs.first().expect("non-empty").min,
                max: runs.last().expect("non-empty").max,
            }));
        }
        Some(Self { runs })
    }

    /// Builds a filter from arbitrary integer keys, sorting and deduplicating
    /// before coalescing contiguous values into runs. Collapses to a single
    /// bounding run when the run count would exceed
    /// [`DIMENSION_KEY_FILTER_MAX_RUNS`]. Returns `None` when `keys` is empty.
    /// Used by callers that already hold a materialized keyset (e.g. a runtime
    /// dynamic-filter payload).
    pub fn from_keys(keys: &[i64]) -> Option<Self> {
        let mut sorted = keys.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        Self::from_sorted_unique_keys(&sorted)
    }

    /// A single-run filter spanning `[min,max]` for callers that only have a
    /// bounding range.
    pub fn from_bounds(bounds: KeyRange) -> Self {
        Self { runs: vec![bounds] }
    }

    /// Inclusive `[min,max]` envelope across every run.
    pub fn bounds(&self) -> KeyRange {
        KeyRange {
            min: self.runs.first().expect("filter has at least one run").min,
            max: self.runs.last().expect("filter has at least one run").max,
        }
    }

    /// Number of disjoint runs (1 for a contiguous/bounding filter).
    pub fn run_count(&self) -> usize {
        self.runs.len()
    }

    /// Whether any run overlaps `candidate` (a fact row group's key statistics).
    pub fn overlaps(&self, candidate: KeyRange) -> bool {
        // Runs are sorted by `min` and disjoint, hence also sorted by `max`, so
        // `run.max < candidate.min` is monotonic over the slice. Find the first
        // run that could reach `candidate`; if it overlaps, the row group is kept,
        // and no later run can either (later runs start strictly higher).
        let index = self.runs.partition_point(|run| run.max < candidate.min);
        self.runs.get(index).is_some_and(|run| ranges_overlap(*run, candidate))
    }
}

/// Accumulates the qualifying dimension key values across decoded dimension
/// batches and coalesces them into a [`DimensionKeyFilter`]. Shared by the
/// in-engine local pass (`derive_dimension_key_filter`) and the out-of-engine
/// resolver, which folds object-store dimension batches through
/// [`Self::observe_batch`] to guarantee identical filter semantics on both sides.
#[derive(Debug, Clone, Default)]
pub struct DimensionKeyAccumulator {
    keys: BTreeSet<i64>,
}

impl DimensionKeyAccumulator {
    /// Adds one decoded batch to the key set, or returns `None` when predicate
    /// or key evaluation fails.
    pub fn observe_batch(&mut self, batch: &RecordBatch, key_column: &str, predicate: &Expr) -> Option<()> {
        let key = batch.column(batch.schema().index_of(key_column).ok()?);
        for row in 0..batch.num_rows() {
            if !eval_bool(predicate, batch, row)? {
                continue;
            }
            let Some(value) = array_i64_value(key.as_ref(), row)? else {
                continue;
            };
            self.keys.insert(value);
        }
        Some(())
    }

    /// Coalesces the collected keys into a filter, or `None` when no key
    /// qualified (an empty dimension key set).
    pub fn finish(&self) -> Option<DimensionKeyFilter> {
        let sorted_unique_keys = self.keys.iter().copied().collect::<Vec<_>>();
        DimensionKeyFilter::from_sorted_unique_keys(&sorted_unique_keys)
    }
}

/// Single-batch wrapper around [`DimensionKeyAccumulator`].
///
/// Returns `None` when predicate/key evaluation fails; otherwise returns the
/// optional filter built from qualifying keys.
pub fn dimension_key_filter_from_batch(
    batch: &RecordBatch,
    key_column: &str,
    predicate: &Expr,
) -> Option<Option<DimensionKeyFilter>> {
    let mut accumulator = DimensionKeyAccumulator::default();
    accumulator.observe_batch(batch, key_column, predicate)?;
    Some(accumulator.finish())
}

/// Selects the fact row groups whose key statistics overlap `filter`, applying
/// the same conservative rules as the in-engine pass: row groups with unknown
/// statistics are retained, and pruning is only reported as applied when it
/// strictly narrows a non-empty selection.
///
/// `row_group_key_ranges` carries one entry per candidate row group in selection
/// order; `None` means the key statistics were unavailable for that row group.
pub(crate) fn select_overlapping_row_groups(
    row_group_key_ranges: &[Option<KeyRange>],
    filter: &DimensionKeyFilter,
) -> RowGroupOverlapSelection {
    let before = row_group_key_ranges.len();
    let mut saw_statistics = false;
    let mut kept = Vec::with_capacity(before);
    for (index, range) in row_group_key_ranges.iter().enumerate() {
        match range {
            Some(range) => {
                saw_statistics = true;
                if filter.overlaps(*range) {
                    kept.push(index);
                }
            },
            None => kept.push(index),
        }
    }
    let applied = saw_statistics && !kept.is_empty() && kept.len() < before;
    RowGroupOverlapSelection { kept, before, saw_statistics, applied }
}

/// Result of [`select_overlapping_row_groups`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RowGroupOverlapSelection {
    /// Indices (into the input order) of row groups to retain.
    pub(crate) kept: Vec<usize>,
    /// Total candidate row groups considered.
    pub(crate) before: usize,
    /// Whether any candidate row group exposed usable key statistics.
    pub(crate) saw_statistics: bool,
    /// Whether pruning strictly narrowed a non-empty selection.
    pub(crate) applied: bool,
}

fn ranges_overlap(left: KeyRange, right: KeyRange) -> bool {
    left.max >= right.min && right.max >= left.min
}
