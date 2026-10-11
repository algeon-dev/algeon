//! Single-pass versus segmented execution of aggregates with `FILTER` clauses.
//!
//! A filtered aggregate evaluates each distinct filter predicate into a boolean
//! mask that stays resident beside the input while the aggregate runs. When the
//! input plus its masks would take too much of the operator budget, the input
//! is cut into row segments and each segment is aggregated into partial state.
//! This policy decides the threshold and the segment size; `filtered_global`
//! and `filtered_grouped` own the execution and report scope-specific facts.

use std::cmp;

use arrow::datatypes::SchemaRef as ArrowSchemaRef;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::exec::operators::aggregate) enum FilteredAggregateScope {
    Global,
    Grouped,
}

/// Internal policy for filtered aggregate materialization and segmentation.
///
/// Filtered global and grouped aggregates use the same memory model but report
/// separate scope-specific facts. Keep these thresholds internal until reports
/// show a stable production tuning contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::exec::operators::aggregate) struct FilteredAggregateSegmentationPolicy {
    single_pass_budget_fraction_denominator: u64,
    segment_target_temporary_bytes: u64,
    min_segment_rows: u64,
    filter_predicate_bytes_per_row: u64,
    segment_budget_divisor: u64,
}

impl FilteredAggregateSegmentationPolicy {
    pub(in crate::exec::operators::aggregate) const DEFAULT_SINGLE_PASS_BUDGET_FRACTION_DENOMINATOR: u64 = 2;
    pub(in crate::exec::operators::aggregate) const DEFAULT_SEGMENT_TARGET_TEMPORARY_BYTES: u64 = 512 * 1024 * 1024;
    pub(in crate::exec::operators::aggregate) const DEFAULT_MIN_SEGMENT_ROWS: u64 = 1;
    pub(in crate::exec::operators::aggregate) const DEFAULT_FILTER_PREDICATE_BYTES_PER_ROW: u64 = 2;
    pub(in crate::exec::operators::aggregate) const DEFAULT_SEGMENT_BUDGET_DIVISOR: u64 = 8;
    pub(in crate::exec::operators::aggregate) const GLOBAL_TEST_SEGMENT_TARGET_TEMPORARY_BYTES_FACT: &str =
        "__algeon_filtered_global_segment_target_bytes";
    pub(in crate::exec::operators::aggregate) const GROUPED_TEST_SEGMENT_TARGET_TEMPORARY_BYTES_FACT: &str =
        "__algeon_filtered_grouped_segment_target_bytes";

    /// Materialized input bytes (input columns plus retained predicate masks)
    /// below which the whole filtered aggregate runs in one pass. The fraction
    /// leaves the rest of the budget to the aggregate's own groupby or
    /// reduction state.
    pub(in crate::exec::operators::aggregate) fn single_pass_threshold_bytes(self, budget_bytes: u64) -> u64 {
        budget_bytes / self.single_pass_budget_fraction_denominator
    }

    /// The threshold's fraction as reported in the strategy facts.
    pub(in crate::exec::operators::aggregate) fn single_pass_budget_fraction_label(self) -> String {
        format!("1/{}", self.single_pass_budget_fraction_denominator)
    }

    /// Bytes per input row that the distinct filter predicates hold while the
    /// aggregate runs. A cuDF BOOL8 mask is one byte per row plus a validity
    /// bitmap; the per-predicate constant rounds that up.
    pub(in crate::exec::operators::aggregate) fn retained_filter_predicate_bytes_per_row(
        self,
        unique_filter_predicate_count: usize,
    ) -> u64 {
        u64::try_from(unique_filter_predicate_count)
            .unwrap_or(u64::MAX)
            .saturating_mul(self.filter_predicate_bytes_per_row)
    }

    /// Rows per segment such that one segment's input and masks fit the
    /// temporary target. The minimum keeps a row wider than the target from
    /// yielding an empty segment.
    pub(in crate::exec::operators::aggregate) fn rows_per_segment(
        self,
        max_segment_input_bytes_per_row: u64,
        target_temporary_bytes: u64,
    ) -> u64 {
        cmp::max(self.min_segment_rows, target_temporary_bytes / max_segment_input_bytes_per_row.max(1))
    }

    pub(in crate::exec::operators::aggregate) fn min_segment_rows_as_usize(self) -> usize {
        usize::try_from(self.min_segment_rows).unwrap_or(usize::MAX)
    }

    /// Temporary byte target for one segment: the default, or the scope's
    /// test override carried in the input schema metadata, then capped at a
    /// share of the budget. The override lets tests force segmentation on a
    /// small input without shrinking the whole budget.
    pub(in crate::exec::operators::aggregate) fn segment_target_temporary_bytes(
        self,
        scope: FilteredAggregateScope,
        input_schema: &ArrowSchemaRef,
        budget_bytes: u64,
    ) -> u64 {
        let configured = input_schema
            .metadata()
            .get(scope.segment_target_temporary_bytes_fact_key())
            .and_then(|value| value.parse::<u64>().ok())
            .filter(|bytes| *bytes > 0)
            .unwrap_or(self.segment_target_temporary_bytes);
        self.segment_temporary_budget_bytes(configured, budget_bytes)
    }

    pub(in crate::exec::operators::aggregate) fn segment_temporary_budget_bytes(
        self,
        target_temporary_bytes: u64,
        budget_bytes: u64,
    ) -> u64 {
        cmp::max(1, cmp::min(target_temporary_bytes, cmp::max(1, budget_bytes / self.segment_budget_divisor)))
    }
}

impl FilteredAggregateScope {
    fn segment_target_temporary_bytes_fact_key(self) -> &'static str {
        match self {
            Self::Global => FilteredAggregateSegmentationPolicy::GLOBAL_TEST_SEGMENT_TARGET_TEMPORARY_BYTES_FACT,
            Self::Grouped => FilteredAggregateSegmentationPolicy::GROUPED_TEST_SEGMENT_TARGET_TEMPORARY_BYTES_FACT,
        }
    }
}

impl Default for FilteredAggregateSegmentationPolicy {
    fn default() -> Self {
        Self {
            single_pass_budget_fraction_denominator: Self::DEFAULT_SINGLE_PASS_BUDGET_FRACTION_DENOMINATOR,
            segment_target_temporary_bytes: Self::DEFAULT_SEGMENT_TARGET_TEMPORARY_BYTES,
            min_segment_rows: Self::DEFAULT_MIN_SEGMENT_ROWS,
            filter_predicate_bytes_per_row: Self::DEFAULT_FILTER_PREDICATE_BYTES_PER_ROW,
            segment_budget_divisor: Self::DEFAULT_SEGMENT_BUDGET_DIVISOR,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, sync::Arc};

    use arrow::datatypes::{DataType, Field, Schema};

    use super::*;

    #[test]
    fn segment_target_uses_default_and_budget_cap() {
        let policy = FilteredAggregateSegmentationPolicy::default();
        let schema: ArrowSchemaRef = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, true)]));

        assert_eq!(
            policy.segment_target_temporary_bytes(FilteredAggregateScope::Global, &schema, 32 * 1024 * 1024),
            4 * 1024 * 1024
        );
        assert_eq!(
            policy.segment_target_temporary_bytes(FilteredAggregateScope::Grouped, &schema, u64::MAX),
            FilteredAggregateSegmentationPolicy::DEFAULT_SEGMENT_TARGET_TEMPORARY_BYTES
        );
    }

    #[test]
    fn segment_target_honors_scope_specific_test_override() {
        let policy = FilteredAggregateSegmentationPolicy::default();
        let mut metadata = HashMap::new();
        metadata.insert(
            FilteredAggregateSegmentationPolicy::GROUPED_TEST_SEGMENT_TARGET_TEMPORARY_BYTES_FACT.to_string(),
            "65536".to_string(),
        );
        let schema: ArrowSchemaRef =
            Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, true)]).with_metadata(metadata));

        assert_eq!(policy.segment_target_temporary_bytes(FilteredAggregateScope::Grouped, &schema, 1024 * 1024), 65536);
        assert_eq!(
            policy.segment_target_temporary_bytes(FilteredAggregateScope::Global, &schema, 1024 * 1024),
            128 * 1024
        );
    }

    #[test]
    fn row_and_predicate_sizing_preserve_minimums() {
        let policy = FilteredAggregateSegmentationPolicy::default();

        assert_eq!(policy.retained_filter_predicate_bytes_per_row(3), 6);
        assert_eq!(policy.rows_per_segment(u64::MAX, 1), 1);
        assert_eq!(policy.min_segment_rows_as_usize(), 1);
        assert_eq!(policy.segment_temporary_budget_bytes(64 * 1024 * 1024, 0), 1);
    }
}
