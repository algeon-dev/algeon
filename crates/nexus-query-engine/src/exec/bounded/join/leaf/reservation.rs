use super::{
    ActiveReservation, BoundedExecutionAttempt, GpuDataFrame, MemoryControlPolicy, MemoryReservationRequest, NodeId,
    QueryError, ReservationKind, Result, SourceChunk, SourceChunkBatch, SourceReadReservationContractQuality,
    observe_device_memory_bytes, source_read_batch_reservation_contract, source_read_reservation_contract,
};
use crate::exec::source::{
    attach_row_group_facts_if_absent, parquet_chunk_path_fact, source_read_batch_request, source_read_request,
};

pub(super) fn attach_join_source_read_resource_facts(
    error: QueryError,
    node_id: NodeId,
    chunk: &SourceChunk,
    attempt: usize,
    max_attempts: usize,
) -> QueryError {
    let error = error
        .with_fact_if_absent("memory_boundary", "join_source_read")
        .with_fact_if_absent("stage", "join_source_read")
        .with_fact_if_absent("node_id", node_id)
        .with_fact_if_absent("attempt", attempt)
        .with_fact_if_absent("max_attempts", max_attempts)
        .with_fact_if_absent("chunk_index", chunk.chunk_index)
        .with_fact_if_absent("file_index", chunk.file_index)
        .with_fact_if_absent("path", parquet_chunk_path_fact(chunk));
    attach_row_group_facts_if_absent(error, chunk)
}

pub(super) fn join_source_read_reservation_request(
    policy: MemoryControlPolicy,
    requested_bytes: u64,
    budget_bytes: u64,
    node_id: NodeId,
    chunk: &SourceChunk,
    attempt: usize,
    max_attempts: usize,
) -> MemoryReservationRequest {
    join_source_read_reservation_request_for_boundary(
        policy,
        requested_bytes,
        budget_bytes,
        node_id,
        chunk,
        "join_source_read",
        attempt,
        max_attempts,
    )
}

pub(super) fn join_source_read_batch_reservation_request(
    policy: MemoryControlPolicy,
    budget_bytes: u64,
    node_id: NodeId,
    batch: &SourceChunkBatch,
    attempt: usize,
    max_attempts: usize,
) -> MemoryReservationRequest {
    let contract = source_read_batch_reservation_contract(policy, batch);
    join_source_read_batch_reservation_request_for_boundary(
        contract.requested_bytes,
        budget_bytes,
        node_id,
        batch,
        "join_source_read",
        attempt,
        max_attempts,
    )
    .with_source_read_contract(contract.quality.as_str(), contract.planning_bytes, contract.estimated_bytes)
}

#[expect(
    clippy::too_many_arguments,
    reason = "bounded execution passes operator-occurrence inputs explicitly beside the attempt state"
)]
pub(super) fn join_source_read_reservation_request_for_boundary(
    policy: MemoryControlPolicy,
    requested_bytes: u64,
    budget_bytes: u64,
    node_id: NodeId,
    chunk: &SourceChunk,
    memory_boundary: &'static str,
    attempt: usize,
    max_attempts: usize,
) -> MemoryReservationRequest {
    let request = source_read_request(
        ReservationKind::SourceRead,
        memory_boundary,
        requested_bytes,
        budget_bytes,
        node_id,
        chunk,
        attempt,
        max_attempts,
    );
    let contract = source_read_reservation_contract(policy, chunk);
    request.with_source_read_contract(contract.quality.as_str(), contract.planning_bytes, contract.estimated_bytes)
}

fn join_source_read_batch_reservation_request_for_boundary(
    requested_bytes: u64,
    budget_bytes: u64,
    node_id: NodeId,
    batch: &SourceChunkBatch,
    memory_boundary: &'static str,
    attempt: usize,
    max_attempts: usize,
) -> MemoryReservationRequest {
    let first_chunk = batch.chunks.first().expect("source chunk batch is non-empty");
    if batch.is_singleton() {
        return join_source_read_reservation_request_for_boundary(
            MemoryControlPolicy::Unbounded,
            requested_bytes,
            budget_bytes,
            node_id,
            first_chunk,
            memory_boundary,
            attempt,
            max_attempts,
        );
    }

    source_read_batch_request(
        ReservationKind::SourceRead,
        memory_boundary,
        requested_bytes,
        budget_bytes,
        node_id,
        batch,
        attempt,
        max_attempts,
    )
}

pub(super) fn observed_join_source_read_reservation_request(
    budget_bytes: u64,
    node_id: NodeId,
    chunk: &SourceChunk,
    attempt: usize,
    max_attempts: usize,
) -> MemoryReservationRequest {
    source_read_request(
        ReservationKind::SourceRead,
        "join_source_read",
        0,
        budget_bytes,
        node_id,
        chunk,
        attempt,
        max_attempts,
    )
    .with_source_read_contract(
        SourceReadReservationContractQuality::ObservedGrowShrink.as_str(),
        chunk.estimated_chunk_bytes,
        chunk.estimated_source_read_bytes,
    )
}

fn observed_join_source_read_batch_reservation_request(
    budget_bytes: u64,
    node_id: NodeId,
    batch: &SourceChunkBatch,
    attempt: usize,
    max_attempts: usize,
) -> MemoryReservationRequest {
    let first_chunk = batch.chunks.first().expect("source chunk batch is non-empty");
    if batch.is_singleton() {
        return observed_join_source_read_reservation_request(
            budget_bytes,
            node_id,
            first_chunk,
            attempt,
            max_attempts,
        );
    }
    let contract = source_read_batch_reservation_contract(MemoryControlPolicy::Unbounded, batch);
    join_source_read_batch_reservation_request_for_boundary(
        0,
        budget_bytes,
        node_id,
        batch,
        "join_source_read",
        attempt,
        max_attempts,
    )
    .with_source_read_contract(
        SourceReadReservationContractQuality::ObservedGrowShrink.as_str(),
        contract.planning_bytes,
        contract.estimated_bytes,
    )
}

pub(super) fn grow_join_source_reservation_to_observed_frame(
    execution: &mut BoundedExecutionAttempt<'_>,
    frame: &GpuDataFrame,
    reservation: &mut ActiveReservation,
    node_id: NodeId,
    chunk: &SourceChunk,
    attempt: usize,
    max_attempts: usize,
) -> Result<()> {
    let observed_bytes = observe_device_memory_bytes(frame, execution.metrics_mut())?;
    if observed_bytes <= reservation.bytes() {
        execution.shrink_to_observed_with_request(
            reservation,
            observed_bytes,
            observed_join_source_read_reservation_request(
                execution.reservations().budget(),
                node_id,
                chunk,
                attempt,
                max_attempts,
            ),
        );
        return Ok(());
    }
    let delta = observed_bytes.saturating_sub(reservation.bytes());
    execution.try_grow(
        reservation,
        observed_join_source_read_reservation_request(
            execution.reservations().budget(),
            node_id,
            chunk,
            attempt,
            max_attempts,
        )
        .with_requested_bytes(delta),
    )
}

pub(super) fn grow_join_source_batch_reservation_to_observed_frame(
    execution: &mut BoundedExecutionAttempt<'_>,
    frame: &GpuDataFrame,
    reservation: &mut ActiveReservation,
    node_id: NodeId,
    batch: &SourceChunkBatch,
    attempt: usize,
    max_attempts: usize,
) -> Result<()> {
    let observed_bytes = observe_device_memory_bytes(frame, execution.metrics_mut())?;
    let observed_request = observed_join_source_read_batch_reservation_request(
        execution.reservations().budget(),
        node_id,
        batch,
        attempt,
        max_attempts,
    );
    if observed_bytes <= reservation.bytes() {
        execution.shrink_to_observed_with_request(reservation, observed_bytes, observed_request);
        return Ok(());
    }
    let delta = observed_bytes.saturating_sub(reservation.bytes());
    execution.try_grow(reservation, observed_request.with_requested_bytes(delta))
}
