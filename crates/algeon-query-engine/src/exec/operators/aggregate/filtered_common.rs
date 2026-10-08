use std::cmp;

use arrow::datatypes::DataType as ArrowDataType;
use cudf::ops::materialization;
use query_runtime::observability::BudgetWindowPeaks;

use super::{filtered_segment_policy::FilteredAggregateSegmentationPolicy, *};
use crate::datatypes::arrow_fixed_width_byte_size;

/// True when `spec` holds an aggregate whose state the row-chunked filtered paths cannot merge:
/// DISTINCT, NUnique, MEDIAN, variance or standard deviation.
pub(super) fn unsupported_stateful_filtered_merge(spec: &AggregateSpec) -> bool {
    spec.aggr_exprs.iter().any(|expr| {
        expr.distinct
            || matches!(
                expr.function,
                AggregateFunction::NUnique
                    | AggregateFunction::Median
                    | AggregateFunction::StddevSamp
                    | AggregateFunction::StddevPop
                    | AggregateFunction::VarSamp
                    | AggregateFunction::VarPop
            )
    })
}

/// Raises the observed temporary and retained peaks to the maxima of a closed budget window.
pub(super) fn add_segmented_budgeted_execution_stats(
    observed_temporary_peak_bytes: &mut u64,
    observed_retained_peak_bytes: &mut u64,
    peaks: BudgetWindowPeaks,
) {
    if let Some(bytes) = peaks.observed_peak_bytes {
        *observed_temporary_peak_bytes = (*observed_temporary_peak_bytes).max(bytes);
    }
    if let Some(bytes) = peaks.retained_bytes {
        *observed_retained_peak_bytes = (*observed_retained_peak_bytes).max(bytes);
    }
}

/// Estimated bytes of one column of `rows` rows: values at the fixed element width (other types
/// at `STRING_OFFSET_BYTES_CONSERVATIVE`), a validity mask when nullable, and the per-column
/// alignment padding. Saturates on overflow.
pub(super) fn estimate_column_bytes(data_type: &ArrowDataType, nullable: bool, rows: u64) -> Result<u64> {
    let width = filtered_fixed_width_dtype_size(data_type).unwrap_or(STRING_OFFSET_BYTES_CONSERVATIVE);
    let validity = if nullable { null_mask_bytes(rows) } else { 0 };
    Ok(rows.saturating_mul(width).saturating_add(validity).saturating_add(PER_COLUMN_ALIGNMENT_PADDING_BYTES))
}

pub(super) fn filtered_fixed_width_dtype_size(data_type: &ArrowDataType) -> Option<u64> {
    arrow_fixed_width_byte_size(data_type)
}

/// How many masked arguments one segment holds, at least one: the segment's temporary-byte
/// target, less the predicate columns retained for the chunk, divided by the bytes of one masked
/// argument over `rows_per_chunk` rows.
pub(super) fn filtered_inputs_per_segment_for_budget(
    max_segment_input_bytes_per_row: u64,
    rows_per_chunk: u64,
    retained_predicate_bytes_per_row: u64,
    budget_bytes: u64,
    target_temporary_bytes: u64,
) -> usize {
    let target = FilteredAggregateSegmentationPolicy::default()
        .segment_temporary_budget_bytes(target_temporary_bytes, budget_bytes);
    let retained_predicate_bytes = retained_predicate_bytes_per_row.saturating_mul(rows_per_chunk);
    let available_for_masked_inputs = target.saturating_sub(retained_predicate_bytes).max(1);
    let per_input_bytes = max_segment_input_bytes_per_row.saturating_mul(rows_per_chunk);
    usize::try_from(cmp::max(1, available_for_masked_inputs / per_input_bytes.max(1))).unwrap_or(usize::MAX)
}

/// cuDF null-mask allocation bytes for `rows`, saturating on overflow like the
/// other terms of the filtered-aggregate footprint estimate.
pub(super) fn null_mask_bytes(rows: u64) -> u64 {
    usize::try_from(rows)
        .ok()
        .and_then(|rows| materialization::null_mask_bytes(rows).ok())
        .map_or(u64::MAX, |bytes| bytes as u64)
}
