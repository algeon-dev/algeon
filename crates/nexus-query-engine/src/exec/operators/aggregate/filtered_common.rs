use std::{cmp, collections::BTreeSet};

use arrow::datatypes::DataType as ArrowDataType;

use super::{filtered_segment_policy::FilteredAggregateSegmentationPolicy, *};
use crate::datatypes::arrow_fixed_width_byte_size;

pub(super) trait FilteredSegmentRuntimeStats {
    fn observed_temporary_peak_bytes_mut(&mut self) -> &mut u64;
    fn observed_retained_peak_bytes_mut(&mut self) -> &mut u64;
    fn unresolved_contracts_mut(&mut self) -> &mut BTreeSet<String>;
}

pub(super) fn unsupported_stateful_filtered_merge(spec: &AggregateSpec) -> bool {
    spec.aggr_exprs.iter().any(|expr| {
        expr.distinct
            || matches!(
                expr.function,
                AggregateFunction::NUnique
                    | AggregateFunction::StddevSamp
                    | AggregateFunction::StddevPop
                    | AggregateFunction::VarSamp
                    | AggregateFunction::VarPop
            )
    })
}

pub(super) fn add_segmented_budgeted_execution_stats<T: FilteredSegmentRuntimeStats>(
    stats: &mut T,
    metrics: &QueryMetrics,
    budget_event_start: usize,
) {
    for event in metrics.budgeted_execution_events.iter().skip(budget_event_start) {
        if let Some(bytes) = event.allocator_high_water_bytes {
            let peak = stats.observed_temporary_peak_bytes_mut();
            *peak = (*peak).max(bytes);
        }
        if let Some(bytes) = event.retained_bytes {
            let peak = stats.observed_retained_peak_bytes_mut();
            *peak = (*peak).max(bytes);
        }
        if let Some(contract) = &event.unresolved_contract {
            stats.unresolved_contracts_mut().insert(contract.clone());
        }
    }
}

pub(super) fn estimate_column_bytes(data_type: &ArrowDataType, nullable: bool, rows: u64) -> Result<u64> {
    let width = filtered_fixed_width_dtype_size(data_type).unwrap_or(STRING_OFFSET_BYTES_CONSERVATIVE);
    let validity = if nullable { null_mask_bytes(rows) } else { 0 };
    Ok(rows.saturating_mul(width).saturating_add(validity).saturating_add(PER_COLUMN_ALIGNMENT_PADDING_BYTES))
}

pub(super) fn filtered_fixed_width_dtype_size(data_type: &ArrowDataType) -> Option<u64> {
    arrow_fixed_width_byte_size(data_type)
}

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

pub(super) fn null_mask_bytes(rows: u64) -> u64 {
    if rows == 0 { 0 } else { rows.div_ceil(8).div_ceil(64) * 64 }
}
