use std::time::Instant;

use parquet::file::reader::FileReader;

use super::{
    ActiveReservation, BoundedExecutionAttempt, BoundedSourceBatchingPolicy, ExecutionContext, GpuDataFrame,
    LocalParquetFile, LocalParquetSource, MemoryControlPolicy, MemoryReservationRequest, NodeId, QueryError,
    QueryMetrics, ReservationKind, ResidentGpuFrame, Result, ScheduledLocalParquetReadOutput, SourceChunk,
    apply_source_scan_limit_to_frame, chunk_retry_facts,
    execution::source_chunk_batch_read_row_groups,
    execution_overrides, is_retryable_chunk_error, observe_device_memory_bytes, plan_source_chunk_batches,
    read_local_parquet_chunk, read_local_parquet_chunk_batch, record_chunk_retry_error,
    reservation_contract::{
        source_read_batch_reservation_request, try_read_local_parquet_chunk_with_corrected_budget_after_error,
    },
    source_adjacent::{
        grow_source_batch_reservation_to_observed, grow_source_reservation_to_observed,
        shrink_source_batch_reservation_to_observed_without_grow, shrink_source_reservation_to_observed_without_grow,
    },
    source_read_reservation_request, split_retryable_chunk,
};

pub(in crate::exec::bounded) fn read_local_parquet_chunks_required_only(
    execution: &mut BoundedExecutionAttempt<'_>,
    source: &LocalParquetSource,
    chunks: &[SourceChunk],
    memory_boundary: &'static str,
) -> Result<Vec<ScheduledLocalParquetReadOutput>> {
    read_local_parquet_chunks_required_only_with_observation(execution, source, chunks, memory_boundary, true)
}

pub(in crate::exec::bounded) fn read_local_parquet_chunks_required_only_with_observation(
    execution: &mut BoundedExecutionAttempt<'_>,
    source: &LocalParquetSource,
    chunks: &[SourceChunk],
    memory_boundary: &'static str,
    grow_source_to_observed: bool,
) -> Result<Vec<ScheduledLocalParquetReadOutput>> {
    read_local_parquet_chunks_serial(execution, source, chunks, memory_boundary, grow_source_to_observed)
}
pub(in crate::exec::bounded) fn concatenate_source_read_outputs(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    mut chunks: Vec<ScheduledLocalParquetReadOutput>,
    concat_output_boundary: &'static str,
    operation: &'static str,
    message: &'static str,
) -> Result<ResidentGpuFrame> {
    if chunks.len() == 1 {
        let ScheduledLocalParquetReadOutput { frame, reservation, .. } =
            chunks.pop().expect("single source chunk checked");
        return Ok(ResidentGpuFrame { frame, reservation });
    }

    let output_reservation_bytes =
        chunks.iter().fold(0u64, |total, chunk| total.saturating_add(chunk.reservation.bytes()));
    let mut output_reservation = match execution.try_reserve(
        MemoryReservationRequest::new(
            ReservationKind::SourceRead,
            output_reservation_bytes,
            execution.reservations().budget(),
        )
        .with_memory_boundary(concat_output_boundary)
        .with_node_id(node_id)
        .with_stage(concat_output_boundary)
        .with_attempt(1, 1),
    ) {
        Ok(reservation) => reservation,
        Err(error) => {
            release_source_read_outputs_reverse(execution, chunks);
            return Err(error);
        },
    };

    let mut input_reservations = Vec::with_capacity(chunks.len());
    let output = (|| {
        let mut frames = Vec::with_capacity(chunks.len());
        for chunk in chunks {
            frames.push(chunk.frame.into_frame());
            input_reservations.push(chunk.reservation);
        }
        let chunk_count = input_reservations.len();
        let first_frame = frames.remove(0);
        let rest_frame_refs = frames.iter().collect::<Vec<_>>();
        let overrides_lease = execution_overrides(execution.context())?;
        let overrides = overrides_lease.overrides();
        let timer = Instant::now();
        let output = first_frame.with_execution(&overrides).concat_many(&rest_frame_refs).map_err(|error| {
            QueryError::from_cudf(operation, message, error)
                .with_fact("node_id", node_id)
                .with_fact("chunk_count", chunk_count)
        })?;
        execution.metrics_mut().operator_time += timer.elapsed();

        let output_frame = GpuDataFrame::new(output);
        let output_bytes = observe_device_memory_bytes(&output_frame, execution.metrics_mut())?;
        if output_bytes > output_reservation.bytes() {
            let delta = output_bytes.saturating_sub(output_reservation.bytes());
            execution.try_grow(
                &mut output_reservation,
                MemoryReservationRequest::new(ReservationKind::SourceRead, delta, execution.reservations().budget())
                    .with_memory_boundary(concat_output_boundary)
                    .with_node_id(node_id)
                    .with_stage(concat_output_boundary)
                    .with_attempt(1, 1),
            )?;
        }
        Ok::<_, QueryError>(output_frame)
    })();
    release_source_reservations_reverse(execution, input_reservations);
    match output {
        Ok(frame) => Ok(ResidentGpuFrame { frame, reservation: output_reservation }),
        Err(error) => {
            execution.release(output_reservation);
            Err(error)
        },
    }
}

fn read_local_parquet_chunks_serial(
    execution: &mut BoundedExecutionAttempt<'_>,
    source: &LocalParquetSource,
    chunks: &[SourceChunk],
    memory_boundary: &'static str,
    grow_source_to_observed: bool,
) -> Result<Vec<ScheduledLocalParquetReadOutput>> {
    if source.has_position_delete_visible_scan_limit() {
        return read_local_parquet_chunks_serial_with_position_delete_visible_limit(
            execution,
            source,
            chunks,
            memory_boundary,
            grow_source_to_observed,
        );
    }

    let batches = generic_source_chunk_batches(execution.context(), source, chunks);
    let mut outputs = Vec::with_capacity(batches.len());
    for batch in batches {
        if let Err(error) =
            execution.context().check_cancellation("nexus_query_engine.exec.bounded.serial_source_chunk")
        {
            release_source_read_outputs_reverse(execution, outputs);
            return Err(error);
        }
        match read_local_parquet_source_batch_or_split(
            execution,
            source,
            batch,
            memory_boundary,
            grow_source_to_observed,
        ) {
            Ok(mut batch_outputs) => outputs.append(&mut batch_outputs),
            Err(error) => {
                release_source_read_outputs_reverse(execution, outputs);
                return Err(error);
            },
        }
    }
    Ok(outputs)
}

fn read_local_parquet_chunks_serial_with_position_delete_visible_limit(
    execution: &mut BoundedExecutionAttempt<'_>,
    source: &LocalParquetSource,
    chunks: &[SourceChunk],
    memory_boundary: &'static str,
    grow_source_to_observed: bool,
) -> Result<Vec<ScheduledLocalParquetReadOutput>> {
    let mut outputs = Vec::with_capacity(chunks.len());
    let mut remaining_rows = source.scan_limit;
    for chunk in chunks {
        if let Err(error) =
            execution.context().check_cancellation("nexus_query_engine.exec.bounded.serial_source_chunk")
        {
            release_source_read_outputs_reverse(execution, outputs);
            return Err(error);
        }
        match read_local_parquet_chunk_required_only(
            execution,
            source,
            chunk,
            memory_boundary,
            grow_source_to_observed,
            false,
            Some(&mut remaining_rows),
        ) {
            Ok(output) => {
                record_position_delete_visible_limit_chunk_metrics(&output.chunk, execution.metrics_mut());
                outputs.push(output);
            },
            Err(error) => {
                release_source_read_outputs_reverse(execution, outputs);
                return Err(error);
            },
        }
        if remaining_rows == Some(0) {
            break;
        }
    }
    Ok(outputs)
}

fn record_position_delete_visible_limit_chunk_metrics(chunk: &SourceChunk, metrics: &mut QueryMetrics) {
    metrics.parquet_files = metrics.parquet_files.saturating_add(1);
    metrics.parquet_bytes_known =
        metrics.parquet_bytes_known.saturating_add(chunk.known_row_group_bytes.or(chunk.known_file_bytes).unwrap_or(0));
    metrics.read_columns = metrics.read_columns.saturating_add(chunk.read_columns.len() as u64);
    if let Some(row_group_count) = chunk.known_read_row_group_count() {
        metrics.row_groups_selected = metrics.row_groups_selected.saturating_add(row_group_count as u64);
    }
}

fn read_local_parquet_source_batch_or_split(
    execution: &mut BoundedExecutionAttempt<'_>,
    source: &LocalParquetSource,
    batch: crate::chunk::SourceChunkBatch,
    memory_boundary: &'static str,
    grow_source_to_observed: bool,
) -> Result<Vec<ScheduledLocalParquetReadOutput>> {
    execution.context().check_cancellation("nexus_query_engine.exec.bounded.serial_source_chunk_batch_retry")?;
    let (max_attempts, min_retry_chunk_row_groups) = source_read_retry_policy(execution.context());
    if batch.is_singleton() {
        let chunk = batch.chunks.first().expect("singleton source batch has one chunk").clone();
        return read_local_parquet_chunk_required_only_or_split(
            execution,
            source,
            chunk,
            memory_boundary,
            grow_source_to_observed,
            1,
            max_attempts,
            min_retry_chunk_row_groups,
        );
    }
    match read_local_parquet_chunk_batch_required_only_no_commit(
        execution,
        source,
        &batch,
        memory_boundary,
        grow_source_to_observed,
    ) {
        Ok(output) => {
            execution
                .metrics_mut()
                .record_source_chunk_commit_count(batch.chunk_count() as u64, batch.row_group_count() as u64);
            Ok(vec![output])
        },
        Err(error) if is_retryable_chunk_error(&error) => {
            record_chunk_retry_error(&error, execution.metrics_mut());
            execution.metrics_mut().record_source_chunk_retry();
            let Some((left, right)) = batch.split_for_retry() else {
                return Err(error);
            };
            crate::exec::retry::backoff_before_retry(
                execution.context(),
                "nexus_query_engine.exec.bounded.source_read_batch_split_retry",
            )?;
            let mut outputs = read_local_parquet_source_batch_or_split(
                execution,
                source,
                left,
                memory_boundary,
                grow_source_to_observed,
            )?;
            match read_local_parquet_source_batch_or_split(
                execution,
                source,
                right,
                memory_boundary,
                grow_source_to_observed,
            ) {
                Ok(mut right_outputs) => {
                    outputs.append(&mut right_outputs);
                    Ok(outputs)
                },
                Err(error) => {
                    release_source_read_outputs_reverse(execution, outputs);
                    Err(error)
                },
            }
        },
        Err(error) => Err(error),
    }
}

fn source_read_retry_policy(context: &ExecutionContext) -> (usize, usize) {
    match context.memory_control_policy() {
        MemoryControlPolicy::Bounded { max_retry_attempts, min_retry_chunk_row_groups, .. } => {
            (max_retry_attempts.max(1), min_retry_chunk_row_groups.max(1))
        },
        MemoryControlPolicy::Unbounded => (1, 1),
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "bounded execution passes operator-occurrence inputs explicitly beside the attempt state"
)]
fn read_local_parquet_chunk_required_only_or_split(
    execution: &mut BoundedExecutionAttempt<'_>,
    source: &LocalParquetSource,
    chunk: SourceChunk,
    memory_boundary: &'static str,
    grow_source_to_observed: bool,
    attempt: usize,
    max_attempts: usize,
    min_retry_chunk_row_groups: usize,
) -> Result<Vec<ScheduledLocalParquetReadOutput>> {
    execution.context().check_cancellation("nexus_query_engine.exec.bounded.serial_source_chunk_retry")?;
    match read_local_parquet_chunk_required_only_attempt(
        execution,
        source,
        &chunk,
        memory_boundary,
        grow_source_to_observed,
        true,
        None,
        attempt,
        max_attempts,
    ) {
        Ok(output) => Ok(vec![output]),
        Err(error) if is_retryable_chunk_error(&error) => {
            record_chunk_retry_error(&error, execution.metrics_mut());
            if attempt < max_attempts
                && let Some((left, right)) = split_retryable_chunk(&chunk, min_retry_chunk_row_groups)
            {
                execution.metrics_mut().record_source_chunk_retry();
                crate::exec::retry::backoff_before_retry(
                    execution.context(),
                    "nexus_query_engine.exec.bounded.serial_source_chunk_split_retry",
                )?;
                let mut outputs = read_local_parquet_chunk_required_only_or_split(
                    execution,
                    source,
                    left,
                    memory_boundary,
                    grow_source_to_observed,
                    attempt + 1,
                    max_attempts,
                    min_retry_chunk_row_groups,
                )?;
                match read_local_parquet_chunk_required_only_or_split(
                    execution,
                    source,
                    right,
                    memory_boundary,
                    grow_source_to_observed,
                    attempt + 1,
                    max_attempts,
                    min_retry_chunk_row_groups,
                ) {
                    Ok(mut right_outputs) => {
                        outputs.append(&mut right_outputs);
                        Ok(outputs)
                    },
                    Err(error) => {
                        release_source_read_outputs_reverse(execution, outputs);
                        Err(error)
                    },
                }
            } else {
                Err(chunk_retry_facts(&chunk, execution.metrics_mut()).attach_to_error(error))
            }
        },
        Err(error) => Err(error),
    }
}

fn read_local_parquet_chunk_required_only(
    execution: &mut BoundedExecutionAttempt<'_>,
    source: &LocalParquetSource,
    chunk: &SourceChunk,
    memory_boundary: &'static str,
    grow_source_to_observed: bool,
    _cache_chunks: bool,
    visible_limit_remaining: Option<&mut Option<usize>>,
) -> Result<ScheduledLocalParquetReadOutput> {
    read_local_parquet_chunk_required_only_attempt(
        execution,
        source,
        chunk,
        memory_boundary,
        grow_source_to_observed,
        _cache_chunks,
        visible_limit_remaining,
        1,
        1,
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "bounded execution passes operator-occurrence inputs explicitly beside the attempt state"
)]
fn read_local_parquet_chunk_required_only_attempt(
    execution: &mut BoundedExecutionAttempt<'_>,
    source: &LocalParquetSource,
    chunk: &SourceChunk,
    memory_boundary: &'static str,
    grow_source_to_observed: bool,
    _cache_chunks: bool,
    visible_limit_remaining: Option<&mut Option<usize>>,
    attempt: usize,
    max_attempts: usize,
) -> Result<ScheduledLocalParquetReadOutput> {
    execution.metrics_mut().record_source_chunk_attempt();
    let selected_row_groups_before_read = execution.metrics_mut().row_groups_selected;
    let source_request = source_read_reservation_request(
        execution.context().memory_control_policy(),
        ReservationKind::SourceRead,
        memory_boundary,
        execution.reservations().budget(),
        chunk,
        attempt,
        max_attempts,
    );
    let mut reservation = execution.try_reserve(source_request)?;
    let source_token_budget = execution.reservations().available_bytes_including(&reservation);
    let frame = match read_local_parquet_chunk(
        execution.context(),
        source,
        chunk,
        source_token_budget,
        memory_boundary,
        execution.metrics_mut(),
    ) {
        Ok(frame) => frame,
        Err(error) => {
            match try_read_local_parquet_chunk_with_corrected_budget_after_error(
                execution,
                source,
                chunk,
                &mut reservation,
                memory_boundary,
                &error,
                attempt,
                max_attempts,
            ) {
                Ok(Some(frame)) => frame,
                Ok(None) => {
                    execution.release(reservation);
                    return Err(error);
                },
                Err(retry_error) => {
                    execution.release(reservation);
                    return Err(retry_error);
                },
            }
        },
    };
    let frame = if let Some(remaining_rows) = visible_limit_remaining {
        match apply_source_scan_limit_to_frame(
            execution.context(),
            frame.frame().clone(),
            *remaining_rows,
            execution.metrics_mut(),
        ) {
            Ok((limited, remaining)) => {
                *remaining_rows = remaining;
                // A slice may retain aliases into a checked-out cache frame.
                // Keep its pin until the derived source result is released so
                // an LRU promotion cannot evict the cache allocation early.
                match GpuDataFrame::try_new_with_source_residency(limited, &frame) {
                    Ok(frame) => frame,
                    Err(error) => {
                        execution.release(reservation);
                        return Err(error);
                    },
                }
            },
            Err(error) => {
                execution.release(reservation);
                return Err(error);
            },
        }
    } else {
        frame
    };
    let mut output = ScheduledLocalParquetReadOutput {
        chunk: chunk.clone(),
        frame,
        reservation,
        observed_bytes: 0,
        observed_row_group_count: None,
    };
    output.observed_bytes = match if grow_source_to_observed {
        grow_source_reservation_to_observed(
            execution,
            &output.frame,
            &mut output.reservation,
            &output.chunk,
            memory_boundary,
            attempt,
            max_attempts,
        )
    } else {
        shrink_source_reservation_to_observed_without_grow(
            execution,
            &output.frame,
            &mut output.reservation,
            &output.chunk,
            memory_boundary,
            attempt,
            max_attempts,
        )
    } {
        Ok(observed_bytes) => observed_bytes,
        Err(error) => {
            execution.release(output.reservation);
            return Err(error);
        },
    };
    let committed_row_groups =
        committed_source_chunk_row_groups(chunk, selected_row_groups_before_read, execution.metrics());
    execution.metrics_mut().record_source_chunk_commit(committed_row_groups);
    Ok(output)
}

fn committed_source_chunk_row_groups(
    chunk: &SourceChunk,
    selected_row_groups_before_read: u64,
    metrics: &QueryMetrics,
) -> u64 {
    if let Some(row_groups) = chunk.known_read_row_group_count() {
        return row_groups as u64;
    }
    let observed_delta = metrics.row_groups_selected.saturating_sub(selected_row_groups_before_read);
    if observed_delta > 0 {
        return observed_delta;
    }
    if chunk.reads_all_row_groups && chunk.path.is_file() {
        return parquet::file::reader::SerializedFileReader::try_from(chunk.path.as_path())
            .ok()
            .map(|reader| reader.metadata().num_row_groups() as u64)
            .unwrap_or(0);
    }
    chunk.row_group_count() as u64
}

pub(in crate::exec::bounded) fn read_local_parquet_chunk_batch_required_only_no_commit(
    execution: &mut BoundedExecutionAttempt<'_>,
    source: &LocalParquetSource,
    batch: &crate::chunk::SourceChunkBatch,
    memory_boundary: &'static str,
    grow_source_to_observed: bool,
) -> Result<ScheduledLocalParquetReadOutput> {
    for _chunk in &batch.chunks {
        execution.metrics_mut().record_source_chunk_attempt();
    }
    let selected_row_groups_before_read = execution.metrics_mut().row_groups_selected;
    let mut reservation = execution.try_reserve(source_read_batch_reservation_request(
        execution.context().memory_control_policy(),
        ReservationKind::SourceRead,
        memory_boundary,
        execution.reservations().budget(),
        batch,
        1,
        1,
    ))?;
    let source_token_budget = execution.reservations().available_bytes_including(&reservation);
    let frame = match read_local_parquet_chunk_batch(
        execution.context(),
        source,
        batch,
        source_token_budget,
        memory_boundary,
        0,
        execution.metrics_mut(),
    ) {
        Ok(frame) => frame,
        Err(error) => {
            execution.release(reservation);
            return Err(error);
        },
    };
    let observed_bytes = match if grow_source_to_observed {
        grow_source_batch_reservation_to_observed(execution, &frame, &mut reservation, batch, memory_boundary, 1, 1)
    } else {
        shrink_source_batch_reservation_to_observed_without_grow(
            execution,
            &frame,
            &mut reservation,
            batch,
            memory_boundary,
            1,
            1,
        )
    } {
        Ok(observed_bytes) => observed_bytes,
        Err(error) => {
            execution.release(reservation);
            return Err(error);
        },
    };
    let first_chunk = batch.chunks.first().expect("source chunk batch is non-empty");
    Ok(ScheduledLocalParquetReadOutput {
        chunk: first_chunk.clone(),
        frame,
        reservation,
        observed_bytes,
        observed_row_group_count: source_chunk_batch_read_row_groups(
            batch,
            selected_row_groups_before_read,
            execution.metrics_mut(),
        ),
    })
}

fn generic_source_chunk_batches(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    chunks: &[SourceChunk],
) -> Vec<crate::chunk::SourceChunkBatch> {
    if !generic_source_chunk_batching_supported(source) {
        return singleton_source_chunk_batches(chunks);
    }
    let target_bytes = BoundedSourceBatchingPolicy::default().generic_source_batch_target_bytes(
        context.memory_control_policy(),
        source_chunk_batch_decode_scratch_risk(source, chunks),
    );
    plan_source_chunk_batches(chunks, target_bytes)
}

fn generic_source_chunk_batching_supported(source: &LocalParquetSource) -> bool {
    source.partition_columns.is_empty()
        && !source.files.iter().any(LocalParquetFile::has_row_delete_files)
        && source.scan_limit.is_none()
        && source.reader_predicate.as_ref().is_none_or(|predicate| source.reader_predicate_is_supported(predicate))
}

fn singleton_source_chunk_batches(chunks: &[SourceChunk]) -> Vec<crate::chunk::SourceChunkBatch> {
    chunks
        .iter()
        .enumerate()
        .map(|(batch_index, chunk)| crate::chunk::SourceChunkBatch {
            batch_index,
            chunks: vec![chunk.clone()],
            estimated_source_read_bytes: chunk
                .estimated_source_read_bytes
                .or(chunk.known_row_group_bytes)
                .or(chunk.estimated_chunk_bytes)
                .or(chunk.known_file_bytes),
            source_batching_reason: "source_gate:generic_batching_disabled".to_string(),
        })
        .collect()
}

fn source_chunk_batch_decode_scratch_risk(source: &LocalParquetSource, chunks: &[SourceChunk]) -> bool {
    source.reader_predicate.is_some()
        && chunks.iter().any(|chunk| {
            chunk.known_pure_filter_compressed_bytes.unwrap_or(0) > 0
                || chunk.known_pure_filter_uncompressed_bytes.unwrap_or(0) > 0
        })
}

pub(in crate::exec::bounded) fn release_source_read_outputs_reverse(
    execution: &mut BoundedExecutionAttempt<'_>,
    mut chunks: Vec<ScheduledLocalParquetReadOutput>,
) {
    while let Some(chunk) = chunks.pop() {
        execution.release(chunk.reservation);
    }
}

fn release_source_reservations_reverse(
    execution: &mut BoundedExecutionAttempt<'_>,
    mut active: Vec<ActiveReservation>,
) {
    while let Some(reservation) = active.pop() {
        execution.release(reservation);
    }
}

#[cfg(test)]
mod tests {
    use std::{path::PathBuf, sync::Arc};

    use arrow::datatypes::{DataType, Field, Schema};

    use super::*;
    use crate::{
        chunk::SourceChunkBatch,
        expr::{BinaryOp, Expr, LiteralValue},
        runtime::SourceChunkPlanningMode,
        source::{LocalParquetSource, ParquetSourceLocation, ResolvedParquetSource},
    };

    #[test]
    fn generic_source_batches_cap_decode_scratch_risk_only() {
        let context = ExecutionContext::builder()
            .with_memory_control_policy(MemoryControlPolicy::Bounded {
                device_budget_bytes: 64 * 1024 * 1024 * 1024,
                max_source_chunk_bytes: 32 * 1024 * 1024 * 1024,
                max_row_groups_per_chunk: 1,
                max_retry_attempts: 1,
                min_retry_chunk_row_groups: 1,
            })
            .build()
            .expect("valid bounded context");
        let chunk_bytes = BoundedSourceBatchingPolicy::DEFAULT_SOURCE_BATCH_READ_MAX_PLANNED_BYTES / 2 + 1;
        let chunks = vec![test_chunk(0, chunk_bytes, false), test_chunk(1, chunk_bytes, false)];

        let no_predicate_batches = generic_source_chunk_batches(&context, &test_source(false), &chunks);
        assert_eq!(no_predicate_batches.len(), 1);
        assert_eq!(no_predicate_batches[0].chunk_count(), 2);

        let scratch_risk_chunks = vec![test_chunk(0, chunk_bytes, true), test_chunk(1, chunk_bytes, true)];
        let predicate_batches = generic_source_chunk_batches(&context, &test_source(true), &scratch_risk_chunks);
        assert_eq!(predicate_batches.len(), 2);
        assert!(predicate_batches.iter().all(SourceChunkBatch::is_singleton));
    }

    #[test]
    fn serial_source_chunk_retry_policy_uses_bounded_context() {
        let context = ExecutionContext::builder()
            .with_memory_control_policy(MemoryControlPolicy::Bounded {
                device_budget_bytes: 64 * 1024 * 1024 * 1024,
                max_source_chunk_bytes: 32 * 1024 * 1024 * 1024,
                max_row_groups_per_chunk: 1024,
                max_retry_attempts: 4,
                min_retry_chunk_row_groups: 8,
            })
            .build()
            .expect("valid bounded context");

        assert_eq!(source_read_retry_policy(&context), (4, 8));
    }

    fn test_source(with_reader_predicate: bool) -> LocalParquetSource {
        let schema = Arc::new(Schema::new(vec![Field::new("c0", DataType::Int64, true)]));
        let source = LocalParquetSource::from_files(Vec::new(), schema, vec!["c0".to_string()]);
        if with_reader_predicate {
            return source.with_reader_predicate(Expr::binary(
                Expr::column("c0"),
                BinaryOp::GtEq,
                Expr::literal(LiteralValue::Int64(0)),
            ));
        }
        source
    }

    fn test_chunk(index: usize, bytes: u64, pure_filter_bytes: bool) -> SourceChunk {
        let pure_filter_bytes = if pure_filter_bytes { bytes } else { 0 };
        SourceChunk {
            source_node_id: NodeId::from_index_for_internal_use(1),
            chunk_index: index,
            file_index: index,
            location: ParquetSourceLocation::LocalPath(PathBuf::from(format!("part-{index}.parquet"))),
            path: PathBuf::from(format!("part-{index}.parquet")),
            resolved_source: ResolvedParquetSource::local_kvikio(format!("part-{index}.parquet")),
            stable_id: format!("file-{index}"),
            row_group_indexes: vec![0],
            reads_all_row_groups: false,
            known_file_bytes: Some(bytes),
            known_row_group_bytes: Some(bytes),
            known_projected_compressed_bytes: Some(bytes),
            known_projected_uncompressed_bytes: Some(bytes),
            known_pure_filter_compressed_bytes: Some(pure_filter_bytes),
            known_pure_filter_uncompressed_bytes: Some(pure_filter_bytes),
            estimated_chunk_bytes: Some(bytes),
            estimated_source_read_bytes: Some(bytes),
            exact_source_read_preflight_bytes: Some(bytes),
            source_file_row_groups: None,
            source_file_row_group_bytes: Some(vec![bytes]),
            source_file_projected_compressed_bytes: Some(vec![bytes]),
            source_file_projected_uncompressed_bytes: Some(vec![bytes]),
            source_file_pure_filter_compressed_bytes: Some(vec![pure_filter_bytes]),
            source_file_pure_filter_uncompressed_bytes: Some(vec![pure_filter_bytes]),
            source_file_row_group_rows: Some(vec![1]),
            reader_decimal_types: Vec::new(),
            reader_decimal_types_footer_verified: false,
            read_columns: vec!["c0".to_string()],
            partition_values: Vec::new(),
            zero_column_count: false,
            known_row_count: Some(1),
            scan_limit_rows: None,
            size_estimate_based: false,
            source_batching_reason: None,
            source_chunk_planning_mode: SourceChunkPlanningMode::Conservative,
            source_chunk_planning_reason: "test_policy".to_string(),
        }
    }
}
