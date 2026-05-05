use arrow::{array::ArrayRef, compute::cast, datatypes::SchemaRef};

use super::*;

pub(super) fn materialize_to_host(
    input: GpuDataFrame,
    output_schema: &SchemaRef,
    metrics: &mut QueryMetrics,
) -> Result<RecordBatch> {
    let timer = Instant::now();
    let output = input.into_frame().to_record_batch().map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.host_arrow.materialize",
            "failed to materialize native cudf output to host Arrow",
            error,
        )
    })?;
    let output = coerce_record_batch_to_schema(output, output_schema)?;
    metrics.materialization_time += timer.elapsed();
    metrics
        .boundary
        .record_device_to_host_materialization(output.num_rows() as u64, record_batch_memory_size(&output) as u64);
    Ok(output)
}

pub(super) fn materialize_bounded_to_host(
    input: CommittedPartial,
    final_node_id: NodeId,
    output_schema: &SchemaRef,
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

    let result = materialize_to_host(input.frame, output_schema, metrics).map_err(|error| {
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

fn coerce_record_batch_to_schema(batch: RecordBatch, output_schema: &SchemaRef) -> Result<RecordBatch> {
    if batch.num_columns() != output_schema.fields().len() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.host_arrow.schema",
            "native host materialization column count must match sink schema",
        )
        .with_fact("actual_columns", batch.num_columns())
        .with_fact("expected_columns", output_schema.fields().len()));
    }
    validate_record_batch_column_names(&batch, output_schema)?;

    let columns = output_schema
        .fields()
        .iter()
        .enumerate()
        .map(|(index, field)| {
            let column = batch.column(index);
            if column.data_type() == field.data_type() {
                return Ok(Arc::clone(column));
            }
            cast(column.as_ref(), field.data_type()).map_err(|error| {
                QueryError::new(
                    QueryErrorCode::UnsupportedType,
                    "cudf_query_engine.host_arrow.schema",
                    "native host materialization could not cast a column to the sink schema type",
                )
                .with_fact("column", field.name())
                .with_fact("actual_type", format!("{:?}", column.data_type()))
                .with_fact("expected_type", format!("{:?}", field.data_type()))
                .with_fact("arrow_error", error.to_string())
            })
        })
        .collect::<Result<Vec<ArrayRef>>>()?;

    RecordBatch::try_new(Arc::clone(output_schema), columns).map_err(|error| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.host_arrow.schema",
            "native host materialization could not attach the sink schema",
        )
        .with_fact("arrow_error", error.to_string())
    })
}

fn validate_record_batch_column_names(batch: &RecordBatch, output_schema: &SchemaRef) -> Result<()> {
    let batch_schema = batch.schema();
    let actual_names = batch_schema.fields().iter().map(|field| field.name().as_str()).collect::<Vec<_>>();
    let expected_names = output_schema.fields().iter().map(|field| field.name().as_str()).collect::<Vec<_>>();

    if actual_names == expected_names {
        return Ok(());
    }

    let mismatch_index =
        actual_names.iter().zip(&expected_names).position(|(actual, expected)| actual != expected).unwrap_or(0);
    Err(QueryError::new(
        QueryErrorCode::InvalidPlan,
        "cudf_query_engine.host_arrow.schema",
        "native host materialization column names must match sink schema order",
    )
    .with_fact("mismatch_index", mismatch_index)
    .with_fact("actual_column", actual_names[mismatch_index])
    .with_fact("expected_column", expected_names[mismatch_index])
    .with_fact("actual_column_names", actual_names.join(","))
    .with_fact("expected_column_names", expected_names.join(",")))
}

pub(super) fn host_materialization_failure_facts(metrics: &QueryMetrics) -> RetryFacts {
    RetryFacts::new()
        .with_resume_boundary("host_materialization")
        .with_committed_chunks(metrics.source_chunks_committed as usize)
        .with_committed_partials(metrics.partial_aggregate_count as usize)
}
