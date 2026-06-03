use super::*;

pub(super) fn attach_join_source_read_resource_facts(
    error: QueryError,
    node_id: NodeId,
    chunk: &SourceChunk,
    attempt: usize,
    max_attempts: usize,
) -> QueryError {
    error
        .with_fact_if_absent("memory_boundary", "join_source_read")
        .with_fact_if_absent("stage", "join_source_read")
        .with_fact_if_absent("node_id", node_id)
        .with_fact_if_absent("attempt", attempt)
        .with_fact_if_absent("max_attempts", max_attempts)
        .with_fact_if_absent("chunk_index", chunk.chunk_index)
        .with_fact_if_absent("file_index", chunk.file_index)
        .with_fact_if_absent("path", chunk.path.display())
        .with_fact_if_absent("row_groups", format_row_groups(chunk))
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
    let mut request = MemoryReservationRequest::new(ReservationKind::SourceRead, requested_bytes, budget_bytes)
        .with_memory_boundary(memory_boundary)
        .with_node_id(node_id)
        .with_stage(memory_boundary)
        .with_chunk_index(chunk.chunk_index)
        .with_file_index(chunk.file_index)
        .with_path(chunk.path.display())
        .with_attempt(attempt, max_attempts);
    if !chunk.reads_all_row_groups {
        request = request.with_row_groups(chunk.row_group_indexes.clone());
    }
    let contract = source_read_reservation_contract(policy, chunk);
    request.with_source_read_contract(contract.quality.as_str(), contract.planning_bytes, contract.estimated_bytes)
}

pub(super) fn observed_join_source_read_reservation_request(
    budget_bytes: u64,
    node_id: NodeId,
    chunk: &SourceChunk,
    attempt: usize,
    max_attempts: usize,
) -> MemoryReservationRequest {
    let mut request = MemoryReservationRequest::new(ReservationKind::SourceRead, 0, budget_bytes)
        .with_memory_boundary("join_source_read")
        .with_node_id(node_id)
        .with_stage("join_source_read")
        .with_chunk_index(chunk.chunk_index)
        .with_file_index(chunk.file_index)
        .with_path(chunk.path.display())
        .with_attempt(attempt, max_attempts);
    if !chunk.reads_all_row_groups {
        request = request.with_row_groups(chunk.row_group_indexes.clone());
    }
    request.with_source_read_contract(
        SourceReadReservationContractQuality::ObservedGrowShrink.as_str(),
        chunk.estimated_chunk_bytes,
        chunk.estimated_source_read_bytes,
    )
}
pub(super) fn grow_join_source_reservation_to_observed_frame(
    frame: &GpuDataFrame,
    reservation: &mut ActiveReservation,
    reservations: &mut ReservationTracker,
    node_id: NodeId,
    chunk: &SourceChunk,
    metrics: &mut QueryMetrics,
    attempt: usize,
    max_attempts: usize,
) -> Result<()> {
    let observed_bytes = observe_device_memory_bytes(frame, metrics)?;
    if observed_bytes <= reservation.bytes() {
        reservations.shrink_to_observed_with_request(
            reservation,
            observed_bytes,
            observed_join_source_read_reservation_request(reservations.budget(), node_id, chunk, attempt, max_attempts),
            metrics,
        );
        return Ok(());
    }
    let delta = observed_bytes.saturating_sub(reservation.bytes());
    reservations.try_grow(
        reservation,
        observed_join_source_read_reservation_request(reservations.budget(), node_id, chunk, attempt, max_attempts)
            .with_requested_bytes(delta),
        metrics,
    )
}
