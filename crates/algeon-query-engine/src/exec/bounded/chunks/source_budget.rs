use super::{ExecutionContext, SourceChunkBatch};

/// Whether an optional read of `batch` may start ahead of its consumer now.
///
/// The decoder needs a minimum read budget; a read is refused when that minimum exceeds the
/// source chunk cap or the device budget. When the attempt reports its allocation, the read
/// is also refused unless the space left under the domain's allocation limit holds the batch's
/// planned bytes plus its read scratch (`scratch_limit_bytes`, or the planned bytes again when
/// unset). A batch with no planned bytes passes that second check; a usage that cannot be read
/// refuses.
pub(in crate::exec::bounded) fn source_read_ahead_admitted(
    context: &ExecutionContext,
    batch: &SourceChunkBatch,
    scratch_limit_bytes: Option<u64>,
) -> bool {
    if batch.chunks.iter().all(|chunk| chunk.zero_column_count) {
        return true;
    }
    let policy = context.memory_control_policy();
    let minimum_read_budget = cudf::DataFrame::minimum_parquet_source_read_budget_bytes() as u64;
    if minimum_read_budget > policy.max_source_chunk_bytes || minimum_read_budget > policy.device_budget_bytes {
        return false;
    }
    let Some(planned_bytes) = batch
        .chunks
        .iter()
        .try_fold(0u64, |total, chunk| chunk.planning_bytes().map(|bytes| total.saturating_add(bytes)))
    else {
        return true;
    };
    match context.attempt_allocation_domain_usage() {
        Ok(Some(usage)) => {
            let free_bytes = usage.allocation_limit_bytes.saturating_sub(usage.current_aligned_bytes);
            planned_bytes.saturating_add(scratch_limit_bytes.unwrap_or(planned_bytes)) <= free_bytes
        },
        Ok(None) => true,
        Err(_) => false,
    }
}
