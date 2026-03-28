use super::*;

pub(super) fn reserve_final_sort_output_bytes(
    node_id: NodeId,
    output_bytes: u64,
    attempt: usize,
    max_attempts: usize,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ActiveReservation> {
    let output_request = MemoryReservationRequest::new(ReservationKind::FinalSort, output_bytes, reservations.budget())
        .with_memory_boundary("final_sort_output")
        .with_node_id(node_id)
        .with_stage("final_sort")
        .with_attempt(attempt, max_attempts)
        .with_compiled_task_id(legacy_bounded_final_sort_task_id(node_id, attempt));
    reservations.try_reserve(output_request, metrics)
}

pub(super) fn finalize_final_sort_output_reservation(
    node_id: NodeId,
    frame: GpuDataFrame,
    mut reservation: ActiveReservation,
    attempt: usize,
    max_attempts: usize,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    let output_bytes = observe_device_memory_bytes(&frame, metrics).map_err(|error| {
        final_sort_runtime_facts("final_sort_output", node_id, attempt, max_attempts, metrics, error)
    })?;

    if output_bytes > reservation.bytes() {
        let grow_request = MemoryReservationRequest::new(
            ReservationKind::FinalSort,
            output_bytes - reservation.bytes(),
            reservations.budget(),
        )
        .with_memory_boundary("final_sort_output")
        .with_node_id(node_id)
        .with_stage("final_sort")
        .with_attempt(attempt, max_attempts)
        .with_compiled_task_id(legacy_bounded_final_sort_task_id(node_id, attempt));
        reservations.try_grow(&mut reservation, grow_request, metrics)?;
    }

    Ok(CommittedPartial { frame, reservation })
}

fn legacy_bounded_final_sort_task_id(node_id: NodeId, attempt: usize) -> String {
    format!("legacy_bounded.final_sort.node{node_id}.attempt{attempt}")
}
