use super::{
    ActiveReservation, BoundedExecutionAttempt, CommittedPartial, GpuDataFrame, MemoryReservationRequest, NodeId,
    ReservationKind, Result, final_sort_runtime_facts, observe_device_memory_bytes,
};

pub(super) fn reserve_final_sort_output_bytes(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    output_bytes: u64,
    attempt: usize,
    max_attempts: usize,
) -> Result<ActiveReservation> {
    let output_request =
        MemoryReservationRequest::new(ReservationKind::FinalSort, output_bytes, execution.reservations().budget())
            .with_memory_boundary("final_sort_output")
            .with_node_id(node_id)
            .with_stage("final_sort")
            .with_attempt(attempt, max_attempts)
            .with_task_registration(
                legacy_bounded_final_sort_task_id(node_id, attempt),
                u32::try_from(attempt).unwrap_or(u32::MAX),
                "final_sort",
            );
    execution.try_reserve(output_request)
}

pub(super) fn finalize_final_sort_output_reservation(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    frame: GpuDataFrame,
    mut reservation: ActiveReservation,
    attempt: usize,
    max_attempts: usize,
) -> Result<CommittedPartial> {
    let output_bytes = observe_device_memory_bytes(&frame, execution.metrics_mut()).map_err(|error| {
        final_sort_runtime_facts("final_sort_output", node_id, attempt, max_attempts, execution.metrics_mut(), error)
    })?;

    if output_bytes > reservation.bytes() {
        let grow_request = MemoryReservationRequest::new(
            ReservationKind::FinalSort,
            output_bytes - reservation.bytes(),
            execution.reservations().budget(),
        )
        .with_memory_boundary("final_sort_output")
        .with_node_id(node_id)
        .with_stage("final_sort")
        .with_attempt(attempt, max_attempts)
        .with_task_registration(
            legacy_bounded_final_sort_task_id(node_id, attempt),
            u32::try_from(attempt).unwrap_or(u32::MAX),
            "final_sort",
        );
        execution.try_grow(&mut reservation, grow_request)?;
    }

    Ok(CommittedPartial { frame, reservation })
}

fn legacy_bounded_final_sort_task_id(node_id: NodeId, attempt: usize) -> String {
    format!("legacy_bounded.final_sort.node{node_id}.attempt{attempt}")
}
