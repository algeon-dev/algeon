use super::{
    super::source::{
        ScheduledLocalParquetReadOutput, apply_source_scan_limit_to_frame, read_local_parquet_chunk,
        read_local_parquet_chunk_batch,
    },
    *,
};
use crate::source::{LocalParquetFile, ParquetReadStrategy};

mod execution;
mod retry;
mod source_adjacent;
mod source_budget;
mod source_budget_policy;
mod source_read;

pub(in crate::exec::bounded) use execution::{
    RawSourceInput, execute_bounded_raw_source_chunk, execute_bounded_raw_source_chunk_batch,
};
pub(in crate::exec) use retry::record_chunk_retry_error;
pub(super) use retry::{chunk_retry_facts, is_retryable_chunk_error, split_retryable_chunk};
pub(in crate::exec::bounded) use source_adjacent::{
    bounded_source_adjacent_chunk_batches_for_work_width, bounded_source_adjacent_chunk_batching_enabled,
    execute_source_adjacent_batch_uncommitted, execute_source_adjacent_chunk_projection,
    execute_source_adjacent_chunk_unary,
};
pub(in crate::exec) use source_budget::source_read_budget_correction_from_error;
pub(super) use source_budget::{join_leaf_prefetch_budget_bytes, source_read_budget_bytes};
use source_budget_policy::SourceReadBudgetPolicy;
pub(super) use source_read::{
    concatenate_source_read_outputs, read_local_parquet_chunk_batch_required_only_no_commit,
    read_local_parquet_chunks_required_only, release_source_read_outputs_reverse,
};
