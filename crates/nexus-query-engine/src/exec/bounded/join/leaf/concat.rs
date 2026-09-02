use super::{
    ActiveReservation, BoundedExecutionAttempt, GpuDataFrame, Instant, MemoryReservationRequest, NodeId, QueryError,
    ReservationKind, ResidentGpuFrame, Result, execution_overrides, observe_device_memory_bytes,
};

pub(super) fn concatenate_join_source_chunks(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    mut chunks: Vec<(GpuDataFrame, ActiveReservation)>,
) -> Result<ResidentGpuFrame> {
    if chunks.len() == 1 {
        let (frame, reservation) = chunks.pop().expect("single chunk checked");
        return Ok(ResidentGpuFrame { frame, reservation });
    }

    let output_reservation_bytes =
        chunks.iter().fold(0u64, |total, (_frame, reservation)| total.saturating_add(reservation.bytes()));
    let mut output_reservation = match execution.try_reserve(
        MemoryReservationRequest::new(
            ReservationKind::SourceRead,
            output_reservation_bytes,
            execution.reservations().budget(),
        )
        .with_memory_boundary("join_source_concat_output")
        .with_node_id(node_id)
        .with_stage("join_source_concat_output")
        .with_attempt(1, 1),
    ) {
        Ok(reservation) => reservation,
        Err(error) => {
            release_join_source_chunks_reverse(execution, chunks);
            return Err(error);
        },
    };

    let mut input_reservations = Vec::with_capacity(chunks.len());
    let mut frames = Vec::with_capacity(chunks.len());
    for (frame, reservation) in chunks {
        frames.push(frame.into_frame());
        input_reservations.push(reservation);
    }
    let chunk_count = input_reservations.len();
    let first_frame = frames.remove(0);
    let rest_frame_refs = frames.iter().collect::<Vec<_>>();
    let overrides_lease = execution_overrides(execution.context())?;
    let overrides = overrides_lease.overrides();
    let timer = Instant::now();
    let output = match first_frame.with_execution(&overrides).concat_many(&rest_frame_refs) {
        Ok(output) => output,
        Err(error) => {
            release_join_source_reservations_reverse(execution, input_reservations);
            execution.release(output_reservation);
            return Err(QueryError::from_cudf(
                "nexus_query_engine.bounded.join.source_concat",
                "failed to concatenate bounded join leaf local parquet source chunks",
                error,
            )
            .with_fact("node_id", node_id)
            .with_fact("chunk_count", chunk_count));
        },
    };
    execution.metrics_mut().operator_time += timer.elapsed();

    let output_frame = GpuDataFrame::new(output);
    let output_bytes = match observe_device_memory_bytes(&output_frame, execution.metrics_mut()) {
        Ok(bytes) => bytes,
        Err(error) => {
            release_join_source_reservations_reverse(execution, input_reservations);
            execution.release(output_reservation);
            return Err(error);
        },
    };
    if output_bytes > output_reservation.bytes() {
        let delta = output_bytes.saturating_sub(output_reservation.bytes());
        if let Err(error) = execution.try_grow(
            &mut output_reservation,
            MemoryReservationRequest::new(ReservationKind::SourceRead, delta, execution.reservations().budget())
                .with_memory_boundary("join_source_concat_output")
                .with_node_id(node_id)
                .with_stage("join_source_concat_output")
                .with_attempt(1, 1),
        ) {
            release_join_source_reservations_reverse(execution, input_reservations);
            execution.release(output_reservation);
            return Err(error);
        }
    }
    release_join_source_reservations_reverse(execution, input_reservations);
    Ok(ResidentGpuFrame { frame: output_frame, reservation: output_reservation })
}

pub(super) fn release_join_source_chunks_reverse(
    execution: &mut BoundedExecutionAttempt<'_>,
    mut chunks: Vec<(GpuDataFrame, ActiveReservation)>,
) {
    while let Some((_, reservation)) = chunks.pop() {
        execution.release(reservation);
    }
}

pub(super) fn release_join_source_reservations_reverse(
    execution: &mut BoundedExecutionAttempt<'_>,
    mut active: Vec<ActiveReservation>,
) {
    while let Some(reservation) = active.pop() {
        execution.release(reservation);
    }
}
