use super::{
    super::source::{
        ScheduledLocalParquetRead, ScheduledLocalParquetReadOutput, apply_source_scan_limit_to_frame,
        read_local_parquet_chunk, read_local_parquet_chunk_batch, source_read_batch_request, source_read_request,
    },
    *,
};
use crate::source::{LocalParquetFile, ParquetReadStrategy};

mod execution;
mod reservation_contract;
mod reservation_policy;
mod retry;
mod scheduling;
mod source_adjacent;
mod source_read;

pub(in crate::exec::bounded) use execution::{
    RawSourceInput, execute_bounded_raw_source_chunk, execute_bounded_raw_source_chunk_batch,
};
pub(in crate::exec) use reservation_contract::{
    SourceReadReservationContractQuality, source_read_batch_reservation_contract,
    source_read_budget_correction_from_error, try_grow_source_read_reservation_for_budget_correction,
};
pub(super) use reservation_contract::{
    chunk_requested_bytes, chunk_reservation_request, join_leaf_prefetch_requested_bytes,
    observed_source_read_reservation_request, prefetch_requested_bytes, source_read_reservation_contract,
    source_read_reservation_request,
};
use reservation_policy::BoundedSourceReservationPolicy;
use retry::is_prefetch_chunk_error;
pub(in crate::exec) use retry::record_chunk_retry_error;
pub(super) use retry::{chunk_retry_facts, is_retryable_chunk_error, split_retryable_chunk};
pub(super) use scheduling::execute_bounded_chunks_with_source_scheduler;
pub(in crate::exec::bounded) use source_adjacent::{
    execute_source_adjacent_chunk_projection, execute_source_adjacent_chunk_unary,
};
pub(super) use source_read::{
    concatenate_source_read_outputs, read_local_parquet_chunk_batch_required_only_no_commit,
    read_local_parquet_chunks_required_only, read_local_parquet_chunks_required_only_with_observation,
    release_source_read_outputs_reverse,
};
