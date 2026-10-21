use std::time::Instant;

use query_planning::policy::PlanningPolicy;
use query_runtime::observability::SourceReadFact;

use super::{
    BoundedExecutionAttempt, BoundedSourceBatchingPolicy, ExecutionContext, GpuDataFrame, LocalParquetSource, NodeId,
    QueryError, ResidentGpuFrame, Result, ScheduledLocalParquetReadOutput, SourceChunk, chunk_retry_facts,
    cudf_query_error,
    execution::{reported_source_chunk_read_row_groups, source_chunk_batch_static_read_row_groups},
    is_retryable_chunk_error, observe_device_memory_bytes, plan_source_chunk_batches, read_local_parquet_chunk,
    read_local_parquet_chunk_batch, record_chunk_retry_error, split_retryable_chunk,
};
use crate::{
    chunk::SourceChunkBatch,
    exec::retry::{RetryLimits, backoff_before_retry},
    rapids_backend::rapids_analysis_context,
};

/// Reads every chunk of a generic source on the calling attempt, one batch at
/// a time, with split-and-retry on resource failures. Outputs come back in
/// chunk order. The whole read is serial on the caller's stream; lane
/// expansion applies only to aggregate and join source streams.
pub(in crate::exec::bounded) fn read_local_parquet_chunks_required_only(
    execution: &mut BoundedExecutionAttempt<'_>,
    source: &LocalParquetSource,
    chunks: &[SourceChunk],
    memory_boundary: &'static str,
) -> Result<Vec<ScheduledLocalParquetReadOutput>> {
    read_local_parquet_chunks_serial(execution, source, chunks, memory_boundary)
}

/// Joins chunk frames into one resident frame. A single chunk is returned as
/// is; several are copied into a new frame by `concat_many`, so inputs and
/// output are resident together for the duration of the copy.
pub(in crate::exec::bounded) fn concatenate_source_read_outputs(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    mut chunks: Vec<ScheduledLocalParquetReadOutput>,
    operation: &'static str,
    message: &'static str,
) -> Result<ResidentGpuFrame> {
    if chunks.len() == 1 {
        let ScheduledLocalParquetReadOutput { frame, observed_bytes, .. } =
            chunks.pop().expect("single source chunk checked");
        return Ok(ResidentGpuFrame::with_device_bytes(frame, observed_bytes));
    }
    let output = (|| {
        let mut frames = Vec::with_capacity(chunks.len());
        for chunk in chunks {
            frames.push(chunk.frame.into_frame());
        }
        let chunk_count = frames.len();
        let first_frame = frames.remove(0);
        let rest_frame_refs = frames.iter().collect::<Vec<_>>();
        let exec_runtime = execution.context().runtime()?;
        let overrides = exec_runtime.exec();
        let timer = Instant::now();
        let output = first_frame.on(overrides).concat_many(&rest_frame_refs).map_err(|error| {
            cudf_query_error(operation, message, error)
                .with_fact("node_id", node_id)
                .with_fact("chunk_count", chunk_count)
        })?;
        execution.metrics().add_operator_time(timer.elapsed());

        let output_frame = GpuDataFrame::new(output);
        let output_bytes = observe_device_memory_bytes(&output_frame, execution.metrics())?;
        Ok::<_, QueryError>((output_frame, output_bytes))
    })();
    match output {
        Ok((frame, output_bytes)) => Ok(ResidentGpuFrame::with_device_bytes(frame, output_bytes)),
        Err(error) => Err(error),
    }
}

fn read_local_parquet_chunks_serial(
    execution: &mut BoundedExecutionAttempt<'_>,
    source: &LocalParquetSource,
    chunks: &[SourceChunk],
    memory_boundary: &'static str,
) -> Result<Vec<ScheduledLocalParquetReadOutput>> {
    let batches = generic_source_chunk_batches(execution.context(), source, chunks);
    let mut outputs = Vec::with_capacity(batches.len());
    for batch in batches {
        execution.context().check_cancellation("query_engine.exec.bounded.serial_source_chunk")?;
        match read_local_parquet_source_batch_or_split(execution, source, batch, memory_boundary) {
            Ok(mut batch_outputs) => outputs.append(&mut batch_outputs),
            Err(error) => {
                return Err(error);
            },
        }
    }
    Ok(outputs)
}

/// Reads one batch, or on a retryable failure splits it and reads the halves.
///
/// A batch splits into smaller batches first and only a singleton batch
/// splits its chunk by row group. The chunk attempt counter therefore starts
/// at 1 when a batch degrades to single chunks; batch splitting shrinks the
/// read without consuming the chunk's own retry budget. A batch that cannot
/// split returns the original error.
fn read_local_parquet_source_batch_or_split(
    execution: &mut BoundedExecutionAttempt<'_>,
    source: &LocalParquetSource,
    batch: SourceChunkBatch,
    memory_boundary: &'static str,
) -> Result<Vec<ScheduledLocalParquetReadOutput>> {
    execution.context().check_cancellation("query_engine.exec.bounded.serial_source_chunk_batch_retry")?;
    let RetryLimits { max_attempts, min_split_row_groups: min_retry_chunk_row_groups } =
        RetryLimits::of(execution.context());
    if batch.is_singleton() {
        let chunk = batch.chunks.first().expect("singleton source batch has one chunk").clone();
        return read_local_parquet_chunk_required_only_or_split(
            execution,
            source,
            chunk,
            memory_boundary,
            1,
            max_attempts,
            min_retry_chunk_row_groups,
        );
    }
    match read_local_parquet_chunk_batch_required_only_no_commit(execution, source, &batch, memory_boundary) {
        Ok(output) => {
            execution.record_source_read(SourceReadFact::ChunkCommitted {
                chunk_count: batch.chunk_count() as u64,
                row_groups: batch.row_group_count() as u64,
            });
            Ok(vec![output])
        },
        Err(error) if is_retryable_chunk_error(&error) => {
            record_chunk_retry_error(&error, execution.metrics());
            execution.record_source_read(SourceReadFact::ChunkRetried);
            let Some((left, right)) = batch.split_for_retry() else {
                return Err(error);
            };
            backoff_before_retry(execution.context(), "query_engine.exec.bounded.source_read_batch_split_retry")?;
            let mut outputs = read_local_parquet_source_batch_or_split(execution, source, left, memory_boundary)?;
            match read_local_parquet_source_batch_or_split(execution, source, right, memory_boundary) {
                Ok(mut right_outputs) => {
                    outputs.append(&mut right_outputs);
                    Ok(outputs)
                },
                Err(error) => Err(error),
            }
        },
        Err(error) => Err(error),
    }
}

fn read_local_parquet_chunk_required_only_or_split(
    execution: &mut BoundedExecutionAttempt<'_>,
    source: &LocalParquetSource,
    chunk: SourceChunk,
    memory_boundary: &'static str,
    attempt: usize,
    max_attempts: usize,
    min_retry_chunk_row_groups: usize,
) -> Result<Vec<ScheduledLocalParquetReadOutput>> {
    execution.context().check_cancellation("query_engine.exec.bounded.serial_source_chunk_retry")?;
    match read_local_parquet_chunk_required_only(execution, source, &chunk, memory_boundary) {
        Ok(output) => Ok(vec![output]),
        Err(error) if is_retryable_chunk_error(&error) => {
            record_chunk_retry_error(&error, execution.metrics());
            if attempt < max_attempts
                && let Some((left, right)) = split_retryable_chunk(&chunk, min_retry_chunk_row_groups)
            {
                execution.record_source_read(SourceReadFact::ChunkRetried);
                backoff_before_retry(execution.context(), "query_engine.exec.bounded.serial_source_chunk_split_retry")?;
                let mut outputs = read_local_parquet_chunk_required_only_or_split(
                    execution,
                    source,
                    left,
                    memory_boundary,
                    attempt + 1,
                    max_attempts,
                    min_retry_chunk_row_groups,
                )?;
                match read_local_parquet_chunk_required_only_or_split(
                    execution,
                    source,
                    right,
                    memory_boundary,
                    attempt + 1,
                    max_attempts,
                    min_retry_chunk_row_groups,
                ) {
                    Ok(mut right_outputs) => {
                        outputs.append(&mut right_outputs);
                        Ok(outputs)
                    },
                    Err(error) => Err(error),
                }
            } else {
                Err(chunk_retry_facts(&chunk, execution).attach_to_error(error))
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
) -> Result<ScheduledLocalParquetReadOutput> {
    execution.record_source_read(SourceReadFact::ChunkAttempted);
    let source_token_budget = execution.device_budget_bytes();
    let frame = read_local_parquet_chunk(
        execution.context(),
        source,
        chunk,
        source_token_budget,
        memory_boundary,
        execution.metrics(),
    )?;
    let observed_bytes = observe_device_memory_bytes(&frame, execution.metrics())?;
    let output =
        ScheduledLocalParquetReadOutput { chunk: chunk.clone(), frame, observed_bytes, observed_row_group_count: None };
    let committed_row_groups = reported_source_chunk_read_row_groups(chunk, None);
    execution.record_source_read(SourceReadFact::ChunkCommitted { chunk_count: 1, row_groups: committed_row_groups });
    Ok(output)
}

/// Decodes a whole batch in one reader call without recording a commit.
/// Callers that may still split or retry the batch record the commit
/// themselves once the output is final. The returned output is attributed to
/// the batch's first chunk, the one the retry facts name.
pub(in crate::exec::bounded) fn read_local_parquet_chunk_batch_required_only_no_commit(
    execution: &mut BoundedExecutionAttempt<'_>,
    source: &LocalParquetSource,
    batch: &SourceChunkBatch,
    memory_boundary: &'static str,
) -> Result<ScheduledLocalParquetReadOutput> {
    for _chunk in &batch.chunks {
        execution.record_source_read(SourceReadFact::ChunkAttempted);
    }
    let source_token_budget = execution.device_budget_bytes();
    let frame = match read_local_parquet_chunk_batch(
        execution.context(),
        source,
        batch,
        source_token_budget,
        memory_boundary,
        0,
        execution.metrics(),
    ) {
        Ok(frame) => frame,
        Err(error) => {
            return Err(error);
        },
    };
    let observed_bytes = observe_device_memory_bytes(&frame, execution.metrics())?;
    let first_chunk = batch.chunks.first().expect("source chunk batch is non-empty");
    Ok(ScheduledLocalParquetReadOutput {
        chunk: first_chunk.clone(),
        frame,
        observed_bytes,
        observed_row_group_count: source_chunk_batch_static_read_row_groups(batch),
    })
}

fn generic_source_chunk_batches(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    chunks: &[SourceChunk],
) -> Vec<SourceChunkBatch> {
    if !generic_source_chunk_batching_supported(source) {
        return singleton_source_chunk_batches(chunks);
    }
    let target_bytes = BoundedSourceBatchingPolicy::default().generic_source_batch_target_bytes(
        context.memory_control_policy(),
        source_chunk_batch_decode_scratch_risk(source, chunks),
    );
    plan_source_chunk_batches(chunks, target_bytes)
}

// A scan limit needs no gate here: planning trims the chunk list to the limit, the batch planner
// isolates the trimmed prefix chunk as `ineligible_scan_limit`, and batch validation rejects any
// chunk carrying `scan_limit_rows`.
fn generic_source_chunk_batching_supported(source: &LocalParquetSource) -> bool {
    source.partition_columns.is_empty()
        && source.reader_predicate.as_ref().is_none_or(|predicate| {
            source.reader_predicate_is_supported(predicate, rapids_analysis_context(PlanningPolicy::default()))
        })
}

fn singleton_source_chunk_batches(chunks: &[SourceChunk]) -> Vec<SourceChunkBatch> {
    chunks
        .iter()
        .enumerate()
        .map(|(batch_index, chunk)| SourceChunkBatch {
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

/// A reader predicate over columns the projection does not keep makes the
/// decoder hold those columns as scratch during the read. Batching then
/// multiplies that scratch, so the batch target shrinks when any chunk
/// reports filter-only bytes.
fn source_chunk_batch_decode_scratch_risk(source: &LocalParquetSource, chunks: &[SourceChunk]) -> bool {
    source.reader_predicate.is_some()
        && chunks.iter().any(|chunk| {
            chunk.known_pure_filter_compressed_bytes.unwrap_or(0) > 0
                || chunk.known_pure_filter_uncompressed_bytes.unwrap_or(0) > 0
        })
}

#[cfg(test)]
mod tests {
    use std::{path::PathBuf, sync::Arc};

    use arrow::datatypes::{DataType, Field, Schema};
    use query_planning::{
        execution_contracts::{MemoryControlPolicy, SourceChunkPlanningMode},
        expr::{BinaryOp, Expr, LiteralValue},
        source::{LocalParquetSource, ParquetSourceLocation, ResolvedParquetSource},
    };
    use query_runtime::runtime::StandaloneConfig;

    use super::*;
    use crate::chunk::SourceChunkBatch;

    #[test]
    fn generic_source_batches_cap_decode_scratch_risk_only() {
        let context = ExecutionContext::standalone(StandaloneConfig {
            memory_control_policy: MemoryControlPolicy {
                device_budget_bytes: 64 * 1024 * 1024 * 1024,
                max_source_chunk_bytes: 32 * 1024 * 1024 * 1024,
                max_row_groups_per_chunk: 1,
                max_retry_attempts: 1,
                min_retry_chunk_row_groups: 1,
            },
            ..StandaloneConfig::default()
        })
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
    fn generic_source_chunk_batches_batch_whole_row_groups_ahead_of_scan_limit_prefix() {
        let context = ExecutionContext::standalone(StandaloneConfig {
            memory_control_policy: MemoryControlPolicy {
                device_budget_bytes: 64 * 1024 * 1024 * 1024,
                max_source_chunk_bytes: 32 * 1024 * 1024 * 1024,
                max_row_groups_per_chunk: 1,
                max_retry_attempts: 1,
                min_retry_chunk_row_groups: 1,
            },
            ..StandaloneConfig::default()
        })
        .expect("valid bounded context");
        let source = test_source(false).with_scan_limit(Some(8));
        let mut chunks = (0..4).map(|index| test_chunk(index, 1024, false)).collect::<Vec<_>>();
        chunks[3].scan_limit_rows = Some(5);

        let batches = generic_source_chunk_batches(&context, &source, &chunks);

        let shape = batches
            .iter()
            .map(|batch| {
                let indexes = batch.chunks.iter().map(|chunk| chunk.chunk_index).collect::<Vec<_>>();
                (indexes, batch.source_batching_reason.as_str())
            })
            .collect::<Vec<_>>();
        assert_eq!(shape, vec![(vec![0, 1, 2], "batched"), (vec![3], "ineligible_scan_limit")]);
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
            source_node_id: NodeId::from_index_for_test(1),
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
            estimated_decoded_bytes: None,
            estimated_column_elements: Some(1),
            max_column_elements: u64::MAX,
            exact_source_read_preflight_bytes: Some(bytes),
            source_file_row_groups: None,
            source_file_row_group_bytes: Some(vec![bytes]),
            source_file_projected_compressed_bytes: Some(vec![bytes]),
            source_file_projected_uncompressed_bytes: Some(vec![bytes]),
            source_file_pure_filter_compressed_bytes: Some(vec![pure_filter_bytes]),
            source_file_pure_filter_uncompressed_bytes: Some(vec![pure_filter_bytes]),
            source_file_row_group_decoded_bytes: None,
            source_file_row_group_column_elements: None,
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
