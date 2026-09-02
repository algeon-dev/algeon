use std::time::Instant;

use parquet::file::reader::FileReader;

use super::{
    BoundedExecutionAttempt, BoundedPlan, CommittedPartial, LocalParquetSource, MemoryControlPolicy,
    PartialAggregateInputEvidence, QueryError, QueryMetrics, ReservationKind, Result, ScheduledLocalParquetReadOutput,
    SourceChunk, SourceChunkBatch, aggregate_spec_has_distinct, attach_chunk_resource_facts,
    attach_source_batch_resource_facts, chunk_reservation_request, chunk_retry_facts,
    execute_partial_aggregate_with_budget, execution_budget_token_for_boundary, is_retryable_chunk_error,
    observe_device_memory_bytes, partial_aggregate_source_chunk_fingerprint, read_local_parquet_chunk,
    read_local_parquet_chunk_batch_required_only_no_commit, record_chunk_retry_error,
    reservation_contract::try_read_local_parquet_chunk_with_corrected_budget_after_error,
    source_adjacent::{
        execute_source_adjacent_chunk_stages, grow_source_reservation_to_observed,
        shrink_source_reservation_to_observed_without_grow,
    },
    source_read_reservation_request, split_retryable_chunk,
};

pub(in crate::exec::bounded) struct RawSourceInput {
    pub(in crate::exec::bounded) partial: CommittedPartial,
    pub(in crate::exec::bounded) source_row_group_count: u64,
}

pub(in crate::exec::bounded) fn execute_bounded_chunk_with_retry(
    execution: &mut BoundedExecutionAttempt<'_>,
    bounded: &BoundedPlan<'_>,
    chunk: &SourceChunk,
) -> Result<Vec<CommittedPartial>> {
    let MemoryControlPolicy::Bounded { max_retry_attempts, min_retry_chunk_row_groups, .. } =
        execution.context().memory_control_policy()
    else {
        unreachable!("execute_bounded_chunk_with_retry is called only for bounded policies");
    };

    execute_bounded_chunk_retry_attempt(
        execution,
        bounded,
        chunk.clone(),
        1,
        max_retry_attempts.max(1),
        min_retry_chunk_row_groups.max(1),
    )
}

pub(in crate::exec::bounded) fn execute_bounded_chunk_retry_attempt(
    execution: &mut BoundedExecutionAttempt<'_>,
    bounded: &BoundedPlan<'_>,
    chunk: SourceChunk,
    attempt: usize,
    max_attempts: usize,
    min_retry_chunk_row_groups: usize,
) -> Result<Vec<CommittedPartial>> {
    execution.context().check_cancellation("nexus_query_engine.exec.bounded.chunk_retry")?;
    match execute_bounded_chunk_attempt(execution, bounded, &chunk, attempt, max_attempts) {
        Ok(partial) => Ok(vec![partial]),
        Err(error) if is_retryable_chunk_error(&error) => {
            retry_after_chunk_error(execution, bounded, chunk, error, attempt, max_attempts, min_retry_chunk_row_groups)
        },
        Err(error) => Err(error),
    }
}
pub(in crate::exec::bounded::chunks) fn retry_after_chunk_error(
    execution: &mut BoundedExecutionAttempt<'_>,
    bounded: &BoundedPlan<'_>,
    chunk: SourceChunk,
    error: QueryError,
    attempt: usize,
    max_attempts: usize,
    min_retry_chunk_row_groups: usize,
) -> Result<Vec<CommittedPartial>> {
    record_chunk_retry_error(&error, execution.metrics_mut());
    if aggregate_spec_has_distinct(&bounded.aggregate.0.spec) {
        return Err(chunk_retry_facts(&chunk, execution.metrics_mut()).attach_to_error(error));
    }
    if attempt < max_attempts
        && let Some((left, right)) = split_retryable_chunk(&chunk, min_retry_chunk_row_groups)
    {
        execution.metrics_mut().record_source_chunk_retry();
        crate::exec::retry::backoff_before_retry(
            execution.context(),
            "nexus_query_engine.exec.bounded.source_chunk_split_retry",
        )?;
        let mut partials = execute_bounded_chunk_retry_attempt(
            execution,
            bounded,
            left,
            attempt + 1,
            max_attempts,
            min_retry_chunk_row_groups,
        )?;
        partials.extend(execute_bounded_chunk_retry_attempt(
            execution,
            bounded,
            right,
            attempt + 1,
            max_attempts,
            min_retry_chunk_row_groups,
        )?);
        return Ok(partials);
    }

    Err(chunk_retry_facts(&chunk, execution.metrics_mut()).attach_to_error(error))
}

pub(in crate::exec::bounded::chunks) fn sum_optional_u64(values: impl IntoIterator<Item = Option<u64>>) -> Option<u64> {
    values.into_iter().try_fold(0u64, |total, value| value.map(|value| total.saturating_add(value)))
}

pub(in crate::exec::bounded::chunks) fn release_committed_partials_reverse(
    execution: &mut BoundedExecutionAttempt<'_>,
    mut partials: Vec<CommittedPartial>,
) {
    while let Some(partial) = partials.pop() {
        execution.release(partial.reservation);
    }
}

/// Reads a source chunk with its pre-aggregate filter and projection stages
/// applied, without running a partial aggregate. Returns the raw filtered and
/// projected data with the represented source row-group count.
///
/// Used by the distinct aggregate path where all raw source data must be
/// concatenated before a single groupby can produce a correct COUNT(DISTINCT).
pub(in crate::exec::bounded) fn execute_bounded_raw_source_chunk(
    execution: &mut BoundedExecutionAttempt<'_>,
    bounded: &BoundedPlan<'_>,
    chunk: &SourceChunk,
) -> Result<RawSourceInput> {
    execution.context().check_cancellation("nexus_query_engine.exec.bounded.raw_source_chunk")?;
    let source = read_bounded_chunk_source(execution, bounded.source, chunk, 1, 1, retain_decoded_source(bounded))
        .map_err(|error| {
            attach_chunk_resource_facts(error, chunk, "source_chunk_read", 1, 1, execution.metrics_mut())
        })?;
    let source_row_group_count = reported_source_chunk_read_row_groups(chunk, source.observed_row_group_count);
    let input = CommittedPartial { frame: source.frame, reservation: source.reservation };
    let partial = execute_source_adjacent_chunk_stages(execution, bounded, chunk, input, 1, 1)?;
    Ok(RawSourceInput { partial, source_row_group_count })
}

pub(in crate::exec::bounded) fn execute_bounded_raw_source_chunk_batch(
    execution: &mut BoundedExecutionAttempt<'_>,
    bounded: &BoundedPlan<'_>,
    batch: &SourceChunkBatch,
) -> Result<RawSourceInput> {
    execution.context().check_cancellation("nexus_query_engine.exec.bounded.raw_source_chunk_batch")?;
    let first_chunk = batch.chunks.first().expect("source chunk batch is non-empty");
    let source = read_local_parquet_chunk_batch_required_only_no_commit(
        execution,
        bounded.source,
        batch,
        "source_chunk_read",
        retain_decoded_source(bounded),
    )
    .map_err(|error| {
        attach_source_batch_resource_facts(error, batch, "source_chunk_read", 1, 1, execution.metrics_mut())
    })?;
    let source_row_group_count = reported_source_chunk_batch_read_row_groups(batch, source.observed_row_group_count);
    let input = CommittedPartial { frame: source.frame, reservation: source.reservation };
    let partial = execute_source_adjacent_chunk_stages(execution, bounded, first_chunk, input, 1, 1)?;
    Ok(RawSourceInput { partial, source_row_group_count })
}

pub(in crate::exec::bounded) fn execute_bounded_chunk_attempt(
    execution: &mut BoundedExecutionAttempt<'_>,
    bounded: &BoundedPlan<'_>,
    chunk: &SourceChunk,
    attempt: usize,
    max_attempts: usize,
) -> Result<CommittedPartial> {
    execution.context().check_cancellation("nexus_query_engine.exec.bounded.chunk_attempt")?;
    let source = read_bounded_chunk_source(
        execution,
        bounded.source,
        chunk,
        attempt,
        max_attempts,
        retain_decoded_source(bounded),
    )
    .map_err(|error| {
        attach_chunk_resource_facts(error, chunk, "source_chunk_read", attempt, max_attempts, execution.metrics_mut())
    })?;
    execute_bounded_chunk_operator_from_source(execution, bounded, source, attempt, max_attempts, |_execution| Ok(()))
}

pub(in crate::exec::bounded::chunks) fn retain_decoded_source(bounded: &BoundedPlan<'_>) -> bool {
    !bounded.has_source_adjacent_stages()
}
pub(in crate::exec::bounded::chunks) fn read_bounded_chunk_source(
    execution: &mut BoundedExecutionAttempt<'_>,
    source: &LocalParquetSource,
    chunk: &SourceChunk,
    attempt: usize,
    max_attempts: usize,
    grow_source_to_observed: bool,
) -> Result<ScheduledLocalParquetReadOutput> {
    execution.context().check_cancellation("nexus_query_engine.exec.bounded.source_chunk_read")?;
    execution.metrics_mut().record_source_chunk_attempt();
    let selected_row_groups_before_read = execution.metrics_mut().row_groups_selected;
    let source_request = source_read_reservation_request(
        execution.context().memory_control_policy(),
        ReservationKind::SourceRead,
        "source_chunk_read",
        execution.reservations().budget(),
        chunk,
        attempt,
        max_attempts,
    );
    let mut source_reservation = execution.try_reserve(source_request)?;
    let source_token_budget = execution.reservations().available_bytes_including(&source_reservation);
    let frame = match read_local_parquet_chunk(
        execution.context(),
        source,
        chunk,
        source_token_budget,
        "source_chunk_read",
        execution.metrics_mut(),
    ) {
        Ok(frame) => frame,
        Err(error) => {
            match try_read_local_parquet_chunk_with_corrected_budget_after_error(
                execution,
                source,
                chunk,
                &mut source_reservation,
                "source_chunk_read",
                &error,
                attempt,
                max_attempts,
            ) {
                Ok(Some(frame)) => frame,
                Ok(None) => {
                    execution.release(source_reservation);
                    return Err(error);
                },
                Err(retry_error) => {
                    execution.release(source_reservation);
                    return Err(retry_error);
                },
            }
        },
    };
    let mut output = ScheduledLocalParquetReadOutput {
        chunk: chunk.clone(),
        frame,
        reservation: source_reservation,
        observed_bytes: 0,
        observed_row_group_count: source_chunk_read_row_groups(
            chunk,
            selected_row_groups_before_read,
            execution.metrics_mut(),
        ),
    };
    if grow_source_to_observed {
        output.observed_bytes = grow_source_reservation_to_observed(
            execution,
            &output.frame,
            &mut output.reservation,
            &output.chunk,
            "source_chunk_read",
            attempt,
            max_attempts,
        )?;
    } else {
        output.observed_bytes = match shrink_source_reservation_to_observed_without_grow(
            execution,
            &output.frame,
            &mut output.reservation,
            &output.chunk,
            "source_chunk_read",
            attempt,
            max_attempts,
        ) {
            Ok(bytes) => bytes,
            Err(error) => {
                execution.release(output.reservation);
                return Err(error);
            },
        };
    }
    Ok(output)
}

pub(in crate::exec::bounded::chunks) fn source_chunk_read_row_groups(
    chunk: &SourceChunk,
    selected_row_groups_before_read: u64,
    metrics: &QueryMetrics,
) -> Option<u64> {
    let observed_delta = metrics.row_groups_selected.saturating_sub(selected_row_groups_before_read);
    if observed_delta > 0 {
        return Some(observed_delta);
    }
    source_chunk_static_read_row_groups(chunk)
}

pub(in crate::exec::bounded::chunks) fn source_chunk_static_read_row_groups(chunk: &SourceChunk) -> Option<u64> {
    if let Some(row_groups) = chunk.known_read_row_group_count() {
        return Some(row_groups as u64);
    }
    if chunk.reads_all_row_groups && chunk.path.is_file() {
        return parquet::file::reader::SerializedFileReader::try_from(chunk.path.as_path())
            .ok()
            .map(|reader| reader.metadata().num_row_groups() as u64);
    }
    None
}

pub(in crate::exec::bounded::chunks) fn reported_source_chunk_read_row_groups(
    chunk: &SourceChunk,
    observed_row_group_count: Option<u64>,
) -> u64 {
    observed_row_group_count
        .or_else(|| source_chunk_static_read_row_groups(chunk))
        .unwrap_or_else(|| chunk.row_group_count() as u64)
}

pub(in crate::exec::bounded::chunks) fn reported_source_chunk_batch_read_row_groups(
    batch: &SourceChunkBatch,
    observed_row_group_count: Option<u64>,
) -> u64 {
    observed_row_group_count
        .or_else(|| source_chunk_batch_static_read_row_groups(batch))
        .unwrap_or_else(|| batch.row_group_count() as u64)
}

pub(in crate::exec::bounded::chunks) fn source_chunk_batch_read_row_groups(
    batch: &SourceChunkBatch,
    selected_row_groups_before_read: u64,
    metrics: &QueryMetrics,
) -> Option<u64> {
    let observed_delta = metrics.row_groups_selected.saturating_sub(selected_row_groups_before_read);
    if observed_delta > 0 {
        return Some(observed_delta);
    }
    source_chunk_batch_static_read_row_groups(batch)
}

pub(in crate::exec::bounded::chunks) fn source_chunk_batch_static_read_row_groups(
    batch: &SourceChunkBatch,
) -> Option<u64> {
    batch.chunks.iter().try_fold(0u64, |total, chunk| {
        source_chunk_static_read_row_groups(chunk).map(|count| total.saturating_add(count))
    })
}

pub(in crate::exec::bounded::chunks) fn execute_bounded_chunk_operator_from_source(
    execution: &mut BoundedExecutionAttempt<'_>,
    bounded: &BoundedPlan<'_>,
    source: ScheduledLocalParquetReadOutput,
    attempt: usize,
    max_attempts: usize,
    before_operator: impl FnOnce(&mut BoundedExecutionAttempt<'_>) -> Result<()>,
) -> Result<CommittedPartial> {
    let ScheduledLocalParquetReadOutput {
        chunk,
        frame,
        reservation: source_reservation,
        observed_bytes: _,
        observed_row_group_count,
    } = source;
    if let Err(error) = execution.context().check_cancellation("nexus_query_engine.exec.bounded.chunk_operator") {
        execution.release(source_reservation);
        return Err(error);
    }
    let mut input = CommittedPartial { frame, reservation: source_reservation };
    input = execute_source_adjacent_chunk_stages(execution, bounded, &chunk, input, attempt, max_attempts)?;
    let CommittedPartial { frame, reservation: source_reservation } = input;

    let operator_request = chunk_reservation_request(
        ReservationKind::ChunkOperator,
        "chunk_operator",
        source_reservation.bytes().min(execution.reservations().remaining_bytes()),
        execution.reservations().budget(),
        &chunk,
        attempt,
        max_attempts,
    );
    let operator_reservation = match execution.try_reserve(operator_request) {
        Ok(reservation) => reservation,
        Err(error) => {
            execution.release(source_reservation);
            return Err(error);
        },
    };
    let operator_token_budget = execution.reservations().available_bytes_including(&operator_reservation);
    let operator_token = match execution_budget_token_for_boundary(
        execution.context(),
        operator_token_budget,
        "chunk_operator",
        "nexus_query_engine.chunk_operator.budget_token",
        "failed to create bounded native chunk operator budget token",
    ) {
        Ok(token) => token,
        Err(error) => {
            execution.release(operator_reservation);
            execution.release(source_reservation);
            return Err(attach_chunk_resource_facts(
                error,
                &chunk,
                "chunk_operator",
                attempt,
                max_attempts,
                execution.metrics_mut(),
            ));
        },
    };
    if let Err(error) = before_operator(execution) {
        execution.release(operator_reservation);
        execution.release(source_reservation);
        return Err(error);
    }

    let partial_input_rows = frame.num_rows() as u64;
    let source_chunk_fingerprint = partial_aggregate_source_chunk_fingerprint(std::slice::from_ref(&chunk));
    let source_chunk_row_group_count = reported_source_chunk_read_row_groups(&chunk, observed_row_group_count);
    let input_evidence = PartialAggregateInputEvidence {
        source_chunk_fingerprint,
        source_adjacent_partial: bounded.has_source_adjacent_stages(),
        source_chunk_row_group_count,
        source_chunk_reads_all_row_groups: chunk.reads_all_row_groups,
        source_chunk_known_row_count: chunk.known_row_count,
        source_chunk_estimated_source_read_bytes: chunk.estimated_source_read_bytes,
        source_chunk_size_estimate_based: chunk.size_estimate_based,
        source_adjacent_partition_strategy: "none",
        source_adjacent_partition_count: None,
        source_adjacent_non_empty_partition_count: None,
        source_adjacent_partition_key_count: None,
        source_adjacent_max_partition_fan_in: None,
    };
    let chunk_operator_timer = Instant::now();
    let chunk_output = execute_partial_aggregate_with_budget(
        execution.context(),
        Some(bounded.aggregate_id),
        bounded.aggregate.0,
        bounded.aggregate_input_schema,
        bounded.aggregate.1,
        frame,
        &operator_token,
        Some(input_evidence),
        execution.metrics_mut(),
    );
    execution.metrics_mut().chunk_operator_time += chunk_operator_timer.elapsed();
    let frame = match chunk_output {
        Ok(frame) => frame,
        Err(error) => {
            execution.release(operator_reservation);
            execution.release(source_reservation);
            return Err(attach_chunk_resource_facts(
                error,
                &chunk,
                "chunk_operator",
                attempt,
                max_attempts,
                execution.metrics_mut(),
            ));
        },
    };

    execution.release(operator_reservation);
    execution.release(source_reservation);

    let partial_bytes = observe_device_memory_bytes(&frame, execution.metrics_mut()).map_err(|error| {
        attach_chunk_resource_facts(error, &chunk, "partial_state", attempt, max_attempts, execution.metrics_mut())
    })?;
    let partial_request = chunk_reservation_request(
        ReservationKind::PartialState,
        "partial_state",
        partial_bytes,
        execution.reservations().budget(),
        &chunk,
        attempt,
        max_attempts,
    );
    let partial_reservation = execution.try_reserve(partial_request)?;
    execution.metrics_mut().record_source_chunk_commit(source_chunk_row_group_count);
    execution.metrics_mut().record_partial_aggregate(
        partial_input_rows,
        frame.num_rows() as u64,
        partial_reservation.bytes(),
    );

    Ok(CommittedPartial { frame, reservation: partial_reservation })
}
