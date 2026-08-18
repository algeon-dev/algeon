use arrow::{
    array::ArrayRef,
    compute::cast,
    datatypes::{Field, SchemaRef},
};

use super::*;

pub(super) fn materialize_to_host(input: GpuResultHandle, metrics: &mut QueryMetrics) -> Result<RecordBatch> {
    let timer = Instant::now();
    let output = input.to_host_record_batch()?;
    metrics.materialization_time += timer.elapsed();
    metrics
        .boundary
        .record_device_to_host_materialization(output.num_rows() as u64, record_batch_memory_size(&output) as u64);
    Ok(output)
}

pub(super) fn materialize_bounded_to_host(
    context: &ExecutionContext,
    input: CommittedPartial,
    final_node_id: NodeId,
    executable_schema: &SchemaRef,
    output_schema: &SchemaRef,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<RecordBatch> {
    let input_reserved_bytes = input.reservation.bytes();
    let result = materialize_bounded_frame_to_host(
        context,
        input.frame,
        input_reserved_bytes,
        final_node_id,
        executable_schema,
        output_schema,
        reservations,
        metrics,
    );
    reservations.release(input.reservation, metrics);
    result
}

pub(in crate::exec) fn materialize_bounded_frame_to_host(
    context: &ExecutionContext,
    frame: GpuDataFrame,
    input_reserved_bytes: u64,
    final_node_id: NodeId,
    executable_schema: &SchemaRef,
    output_schema: &SchemaRef,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<RecordBatch> {
    let request =
        MemoryReservationRequest::new(ReservationKind::Materialization, input_reserved_bytes, reservations.budget())
            .with_memory_boundary("host_materialization")
            .with_node_id(final_node_id)
            .with_stage("materialization")
            .with_attempt(1, 1)
            .with_task_registration(
                format!("legacy_bounded.materialization.node{final_node_id}.attempt1"),
                1,
                "materialization",
            );
    let materialization_reservation = match reservations.try_reserve(request, metrics) {
        Ok(reservation) => reservation,
        Err(error) => return Err(host_materialization_failure_facts(metrics).attach_to_error(error)),
    };

    let handle = match GpuResultHandle::new(context, frame, Arc::clone(executable_schema)) {
        Ok(handle) => handle,
        Err(error) => {
            reservations.release(materialization_reservation, metrics);
            return Err(host_materialization_failure_facts(metrics).attach_to_error(error));
        },
    };
    let result = materialize_to_host_with_schema(handle, output_schema, metrics).map_err(|error| {
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
    reservations.release(materialization_reservation, metrics);
    result.map_err(|error| host_materialization_failure_facts(metrics).attach_to_error(error))
}

pub(super) fn materialize_gpu_frame_to_host(
    context: &ExecutionContext,
    frame: GpuDataFrame,
    executable_schema: SchemaRef,
    output_schema: &SchemaRef,
    metrics: &mut QueryMetrics,
) -> Result<RecordBatch> {
    let handle = GpuResultHandle::new(context, frame, executable_schema)?;
    materialize_to_host_with_schema(handle, output_schema, metrics)
}

fn materialize_to_host_with_schema(
    input: GpuResultHandle,
    output_schema: &SchemaRef,
    metrics: &mut QueryMetrics,
) -> Result<RecordBatch> {
    let output = materialize_to_host(input, metrics)?;
    if output.schema().as_ref() == output_schema.as_ref() {
        return Ok(output);
    }
    record_batch_with_sink_schema(output, output_schema)
}

fn record_batch_with_sink_schema(batch: RecordBatch, schema: &SchemaRef) -> Result<RecordBatch> {
    if batch.num_columns() != schema.fields().len() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.host_arrow.materialize_schema",
            "materialized Arrow column count must match host sink schema field count",
        )
        .with_fact("actual_columns", batch.num_columns())
        .with_fact("sink_fields", schema.fields().len()));
    }
    for (index, (materialized_field, sink_field)) in
        batch.schema().fields().iter().zip(schema.fields().iter()).enumerate()
    {
        if sink_field.data_type() != materialized_field.data_type() {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.host_arrow.materialize_schema",
                "materialized Arrow type must match host sink schema by ordinal",
            )
            .with_fact("field_index", index)
            .with_fact("sink_field", sink_field.name())
            .with_fact("materialized_field", materialized_field.name())
            .with_fact("materialized_type", format!("{:?}", materialized_field.data_type()))
            .with_fact("sink_type", format!("{:?}", sink_field.data_type()))
            .with_fact("materialized_nullable", materialized_field.is_nullable())
            .with_fact("sink_nullable", sink_field.is_nullable())
            .with_fact("materialized_metadata", format!("{:?}", materialized_field.metadata()))
            .with_fact("sink_metadata", format!("{:?}", sink_field.metadata())));
        }
    }
    let columns = batch
        .columns()
        .iter()
        .zip(schema.fields().iter())
        .enumerate()
        .map(|(index, (column, field))| column_with_sink_field_contract(column, field.as_ref(), index))
        .collect::<Result<Vec<_>>>()?;
    RecordBatch::try_new(schema.clone(), columns).map_err(|error| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.host_arrow.materialize_schema",
            "failed to attach host sink schema to materialized Arrow output",
        )
        .with_source(error)
    })
}

fn column_with_sink_field_contract(column: &ArrayRef, sink_field: &Field, field_index: usize) -> Result<ArrayRef> {
    if column.data_type() == sink_field.data_type() {
        return Ok(Arc::clone(column));
    }
    cast(column.as_ref(), sink_field.data_type()).map_err(|error| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.host_arrow.materialize_schema",
            "failed to cast materialized Arrow column to host sink schema field type",
        )
        .with_fact("field_index", field_index)
        .with_fact("sink_field", sink_field.name())
        .with_fact("materialized_type", format!("{:?}", column.data_type()))
        .with_fact("sink_type", format!("{:?}", sink_field.data_type()))
        .with_source(error)
    })
}

pub(super) fn host_materialization_failure_facts(metrics: &QueryMetrics) -> RetryFacts {
    RetryFacts::new()
        .with_resume_boundary("host_materialization")
        .with_committed_chunks(metrics.source_chunks_committed as usize)
        .with_committed_partials(metrics.partial_aggregate_count as usize)
}
