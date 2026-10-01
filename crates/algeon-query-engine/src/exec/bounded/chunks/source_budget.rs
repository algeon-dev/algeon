use super::{MemoryControlPolicy, SourceChunk};

/// Whether a join leaf may start an optional prefetch read for `chunk`.
///
/// The decoder needs a minimum read budget; a prefetch is skipped when that
/// minimum exceeds the source chunk cap or the device budget.
pub(in crate::exec::bounded) fn join_leaf_prefetch_admitted(policy: MemoryControlPolicy, chunk: &SourceChunk) -> bool {
    let MemoryControlPolicy { device_budget_bytes, max_source_chunk_bytes, .. } = policy;
    if chunk.zero_column_count {
        return true;
    }
    let minimum_read_budget = cudf::DataFrame::minimum_parquet_source_read_budget_bytes() as u64;
    minimum_read_budget <= max_source_chunk_bytes && minimum_read_budget <= device_budget_bytes
}
