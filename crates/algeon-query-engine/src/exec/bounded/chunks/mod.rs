use super::*;
use crate::{
    exec::source::{ScheduledLocalParquetReadOutput, read_local_parquet_chunk, read_local_parquet_chunk_batch},
    source::ParquetReadStrategy,
};

mod execution;
mod retry;
mod source_adjacent;
mod source_budget;
mod source_read;

pub(in crate::exec::bounded) use execution::{
    RawSourceInput, execute_bounded_raw_source_chunk, execute_bounded_raw_source_chunk_batch,
};
pub(in crate::exec) use retry::record_chunk_retry_error;
pub(super) use retry::{chunk_retry_facts, is_retryable_chunk_error, split_retryable_chunk};
pub(in crate::exec::bounded) use source_adjacent::{
    bounded_source_adjacent_chunk_batches_for_work_width, bounded_source_adjacent_chunk_batching_enabled,
    execute_source_adjacent_batch_uncommitted,
};
pub(super) use source_budget::join_leaf_prefetch_admitted;
pub(super) use source_read::{
    concatenate_source_read_outputs, read_local_parquet_chunk_batch_required_only_no_commit,
    read_local_parquet_chunks_required_only,
};
