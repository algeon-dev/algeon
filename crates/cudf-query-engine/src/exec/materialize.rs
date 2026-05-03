use super::*;

pub(super) fn materialize_to_host(input: GpuDataFrame, metrics: &mut QueryMetrics) -> Result<RecordBatch> {
    let timer = Instant::now();
    let output = input.into_frame().to_record_batch().map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.host_arrow.materialize",
            "failed to materialize native cudf output to host Arrow",
            error,
        )
    })?;
    metrics.materialization_time += timer.elapsed();
    metrics
        .boundary
        .record_device_to_host_materialization(output.num_rows() as u64, record_batch_memory_size(&output) as u64);
    Ok(output)
}

pub(super) fn materialize_bounded_to_host(
    input: CommittedPartial,
    final_node_id: NodeId,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<RecordBatch> {
    let request = MemoryReservationRequest::new(
        ReservationKind::Materialization,
        input.reservation.bytes(),
        reservations.budget(),
    )
    .with_memory_boundary("host_materialization")
    .with_node_id(final_node_id)
    .with_stage("materialization")
    .with_attempt(1, 1);
    let materialization_reservation = match reservations.try_reserve(request, metrics) {
        Ok(reservation) => reservation,
        Err(error) => {
            reservations.release(input.reservation);
            return Err(host_materialization_failure_facts(metrics).attach_to_error(error));
        },
    };

    let result = materialize_to_host(input.frame, metrics).map_err(|error| {
        RuntimeResourceFacts {
            memory_boundary: "host_materialization",
            stage: "materialization",
            node_id: final_node_id,
            attempt: 1,
            max_attempts: 1,
            resume_boundary: "host_materialization",
            metrics,
        }
        .attach_to(error)
    });
    reservations.release(materialization_reservation);
    reservations.release(input.reservation);
    result.map_err(|error| host_materialization_failure_facts(metrics).attach_to_error(error))
}

pub(super) fn host_materialization_failure_facts(metrics: &QueryMetrics) -> RetryFacts {
    RetryFacts::new()
        .with_resume_boundary("host_materialization")
        .with_committed_chunks(metrics.source_chunks_committed as usize)
        .with_committed_partials(metrics.partial_aggregate_count as usize)
}
