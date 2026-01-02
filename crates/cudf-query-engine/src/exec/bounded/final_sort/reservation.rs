use super::*;

pub(super) fn reserve_final_sort_output_bytes(
    bounded: &BoundedPlan<'_>,
    output_bytes: u64,
    attempt: usize,
    max_attempts: usize,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ActiveReservation> {
    let output_request = MemoryReservationRequest::new(ReservationKind::FinalSort, output_bytes, reservations.budget())
        .with_memory_boundary("final_sort_output")
        .with_node_id(bounded.aggregate_id)
        .with_stage("final_sort")
        .with_attempt(attempt, max_attempts);
    reservations.try_reserve(output_request, metrics)
}

pub(super) fn finalize_final_sort_output_reservation(
    bounded: &BoundedPlan<'_>,
    frame: GpuDataFrame,
    mut reservation: ActiveReservation,
    attempt: usize,
    max_attempts: usize,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    let output_bytes = observe_device_memory_bytes(&frame, metrics).map_err(|error| {
        RuntimeResourceFacts {
            memory_boundary: "final_sort_output",
            stage: "final_sort",
            node_id: bounded.aggregate_id,
            attempt,
            max_attempts,
            resume_boundary: "final_sort",
            metrics,
        }
        .attach_to(error)
    })?;

    if output_bytes > reservation.bytes() {
        let grow_request = MemoryReservationRequest::new(
            ReservationKind::FinalSort,
            output_bytes - reservation.bytes(),
            reservations.budget(),
        )
        .with_memory_boundary("final_sort_output")
        .with_node_id(bounded.aggregate_id)
        .with_stage("final_sort")
        .with_attempt(attempt, max_attempts);
        reservations.try_grow(&mut reservation, grow_request, metrics)?;
    }

    Ok(CommittedPartial { frame, reservation })
}
