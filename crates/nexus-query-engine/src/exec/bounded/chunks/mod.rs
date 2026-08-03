use std::collections::VecDeque;

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

// Private re-export so sibling submodules can reach `is_prefetch_chunk_error`
// via `use super::*;` without exposing it outside the chunks module.
// Seed the chunks namespace so sibling submodules resolve shared helpers via `use super::*;`.
// Restricted to this module tree; external visibility is restored by the re-exports below.
#[allow(unused_imports)] // sibling resolution via use super::*
pub(in crate::exec::bounded::chunks) use execution::*;
// Re-exports preserving original external visibility.
pub(in crate::exec::bounded) use execution::{
    RawSourceInput, execute_bounded_raw_source_chunk, execute_bounded_raw_source_chunk_batch,
};
// Keep attempt/retry_attempt names in the chunks public surface for symmetry with the interrupt call
// sites, even though only the root retry entry is referenced from outside the tree today.
#[allow(unused_imports)]
pub(super) use execution::{
    execute_bounded_chunk_attempt, execute_bounded_chunk_retry_attempt, execute_bounded_chunk_with_retry,
};
#[allow(unused_imports)] // sibling resolution via use super::*
pub(in crate::exec::bounded::chunks) use reservation_contract::*;
#[allow(unused_imports)]
pub(in crate::exec) use reservation_contract::{
    SourceReadBudgetCorrection, SourceReadReservationContract, SourceReadReservationContractQuality,
    observed_source_read_batch_reservation_request, source_chunk_batch_reservation_request,
    source_read_batch_reservation_contract, source_read_batch_reservation_request,
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
#[allow(unused_imports)] // sibling resolution via use super::*
pub(in crate::exec::bounded::chunks) use source_adjacent::*;
pub(in crate::exec::bounded) use source_adjacent::{
    execute_source_adjacent_chunk_projection, execute_source_adjacent_chunk_unary,
};
pub(super) use source_read::{
    concatenate_source_read_outputs, read_local_parquet_chunk_batch_required_only_no_commit,
    read_local_parquet_chunks_required_only, read_local_parquet_chunks_required_only_with_observation,
    release_source_read_outputs_reverse,
};
