use std::time::Instant;

use super::{
    BoundedExecutionAttempt, BoundedPlan, BoundedSourceBatchingPolicy, CommittedPartial, ExecutionBudgetToken,
    ExecutionContext, ExecutionMetrics, GpuDataFrame, LocalParquetFile, LocalParquetSource, ParquetReadStrategy,
    PartialAggregateInputEvidence, ProjectionNode, Result, ScheduledLocalParquetReadOutput, SourceChunk,
    SourceChunkBatch, attach_chunk_resource_facts, execute_filter_with_budget, execute_partial_aggregate_with_budget,
    execute_projection_with_budget, execute_projection_with_budget_and_prepared,
    execution::{execute_bounded_chunk_with_retry, release_committed_partials_reverse, sum_optional_u64},
    execution_budget_token_for_boundary, is_retryable_chunk_error, is_source_adjacent_grouped_aggregate,
    observe_device_memory_bytes, partial_aggregate_source_chunk_fingerprint, plan_source_chunk_batches,
    projection_regexp_replace_prepare, read_local_parquet_chunk_batch_required_only_no_commit,
    record_chunk_retry_error,
};
use crate::observability::SourceReadFact;

pub(in crate::exec::bounded::chunks) fn bounded_source_adjacent_chunk_batching_enabled(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    chunks: &[SourceChunk],
) -> bool {
    bounded.has_source_adjacent_stages()
        && bounded.source.partition_columns.is_empty()
        && !bounded.source.files.iter().any(LocalParquetFile::has_row_delete_files)
        && bounded.source.scan_limit.is_none()
        && bounded.source.reader_predicate.as_ref().is_none_or(|predicate| {
            bounded.source.reader_predicate_is_supported(predicate, crate::rapids_backend::rapids_analysis_context())
        })
        && bounded_source_adjacent_chunk_batches(context, bounded, chunks).iter().any(|batch| batch.chunk_count() > 1)
}

pub(in crate::exec::bounded::chunks) fn bounded_source_adjacent_chunk_batches(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    chunks: &[SourceChunk],
) -> Vec<SourceChunkBatch> {
    if bounded.source.files.iter().any(LocalParquetFile::has_row_delete_files) {
        return singleton_source_chunk_batches(chunks);
    }
    let batching_policy = BoundedSourceBatchingPolicy::default();
    let max_batch_source_read_bytes = batching_policy.source_adjacent_raw_source_batch_target_bytes(
        context.memory_control_policy(),
        source_adjacent_source_requires_remote_direct_read(bounded.source, chunks),
        is_source_adjacent_grouped_aggregate(bounded),
    );
    plan_source_chunk_batches(chunks, max_batch_source_read_bytes)
}

#[cfg(test)]
pub(in crate::exec::bounded::chunks) fn source_adjacent_raw_source_batch_max_bytes(
    source: &LocalParquetSource,
    chunks: &[SourceChunk],
    source_adjacent_grouped_aggregate: bool,
) -> u64 {
    BoundedSourceBatchingPolicy::default().source_adjacent_raw_source_batch_max_bytes(
        source_adjacent_source_requires_remote_direct_read(source, chunks),
        source_adjacent_grouped_aggregate,
    )
}

pub(in crate::exec::bounded::chunks) fn source_adjacent_source_requires_remote_direct_read(
    source: &LocalParquetSource,
    chunks: &[SourceChunk],
) -> bool {
    source.scan_limit.is_none()
        && source.partition_columns.is_empty()
        && chunks.iter().all(source_chunk_requires_remote_direct_read)
}

pub(in crate::exec::bounded::chunks) fn source_chunk_requires_remote_direct_read(chunk: &SourceChunk) -> bool {
    matches!(chunk.resolved_source.read_strategy(), ParquetReadStrategy::RemoteDirect)
}

pub(in crate::exec::bounded::chunks) fn singleton_source_chunk_batches(
    chunks: &[SourceChunk],
) -> Vec<SourceChunkBatch> {
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
            source_batching_reason: "source_gate:position_deletes".to_string(),
        })
        .collect()
}

pub(in crate::exec::bounded::chunks) fn execute_bounded_source_adjacent_chunk_batches_serial(
    execution: &mut BoundedExecutionAttempt<'_>,
    bounded: &BoundedPlan<'_>,
    chunks: &[SourceChunk],
) -> Result<Vec<CommittedPartial>> {
    let batches = bounded_source_adjacent_chunk_batches(execution.context(), bounded, chunks);
    let mut partials = Vec::with_capacity(batches.len());
    for batch in batches {
        execution.context().check_cancellation("nexus_query_engine.exec.bounded.source_adjacent_chunk_batch")?;
        match execute_bounded_source_adjacent_chunk_batch_or_split(execution, bounded, batch) {
            Ok(mut batch_partials) => {
                partials.append(&mut batch_partials);
            },
            Err(error) => {
                release_committed_partials_reverse(partials);
                return Err(error);
            },
        }
    }
    Ok(partials)
}

pub(in crate::exec::bounded::chunks) fn execute_bounded_source_adjacent_chunk_batch_or_split(
    execution: &mut BoundedExecutionAttempt<'_>,
    bounded: &BoundedPlan<'_>,
    batch: SourceChunkBatch,
) -> Result<Vec<CommittedPartial>> {
    execution.context().check_cancellation("nexus_query_engine.exec.bounded.source_adjacent_chunk_batch_retry")?;
    if batch.is_singleton() {
        let chunk = batch.chunks.first().expect("singleton source chunk batch has one child");
        return execute_bounded_chunk_with_retry(execution, bounded, chunk);
    }
    match execute_bounded_source_adjacent_chunk_batch_attempt(execution, bounded, &batch) {
        Ok(partial) => Ok(vec![partial]),
        Err(error) if is_retryable_chunk_error(&error) => {
            record_chunk_retry_error(&error, execution.metrics());
            execution.record_source_read(SourceReadFact::ChunkRetried);
            let Some((left, right)) = batch.split_for_retry() else {
                return Err(error);
            };
            crate::exec::retry::backoff_before_retry(
                execution.context(),
                "nexus_query_engine.exec.bounded.source_adjacent_chunk_batch_split_retry",
            )?;
            let mut partials = execute_bounded_source_adjacent_chunk_batch_or_split(execution, bounded, left)?;
            match execute_bounded_source_adjacent_chunk_batch_or_split(execution, bounded, right) {
                Ok(mut right_partials) => {
                    partials.append(&mut right_partials);
                    Ok(partials)
                },
                Err(error) => {
                    release_committed_partials_reverse(partials);
                    Err(error)
                },
            }
        },
        Err(error) => Err(error),
    }
}

pub(in crate::exec::bounded::chunks) fn execute_bounded_source_adjacent_chunk_batch_attempt(
    execution: &mut BoundedExecutionAttempt<'_>,
    bounded: &BoundedPlan<'_>,
    batch: &SourceChunkBatch,
) -> Result<CommittedPartial> {
    let output =
        read_local_parquet_chunk_batch_required_only_no_commit(execution, bounded.source, batch, "source_chunk_read")?;
    execute_bounded_chunk_batch_operator_from_source(execution, bounded, batch, output)
}

pub(in crate::exec::bounded::chunks) fn execute_bounded_chunk_batch_operator_from_source(
    execution: &mut BoundedExecutionAttempt<'_>,
    bounded: &BoundedPlan<'_>,
    batch: &SourceChunkBatch,
    source: ScheduledLocalParquetReadOutput,
) -> Result<CommittedPartial> {
    let first_chunk = batch.chunks.first().expect("source chunk batch is non-empty");
    let ScheduledLocalParquetReadOutput { frame, observed_bytes, .. } = source;
    execution.context().check_cancellation("nexus_query_engine.exec.bounded.chunk_batch_operator")?;
    let mut input = CommittedPartial::with_device_bytes(frame, observed_bytes);
    input = execute_source_adjacent_chunk_stages(execution, bounded, first_chunk, input, 1, 1)?;
    let CommittedPartial { frame, .. } = input;

    let operator_token_budget = execution.device_budget_bytes();
    let operator_token = match execution_budget_token_for_boundary(
        execution.context(),
        operator_token_budget,
        "chunk_operator",
        "nexus_query_engine.chunk_operator.budget_token",
        "failed to create bounded native chunk batch operator budget token",
    ) {
        Ok(token) => token,
        Err(error) => {
            return Err(attach_chunk_resource_facts(error, first_chunk, "chunk_operator", 1, 1, execution.metrics()));
        },
    };

    let partial_input_rows = frame.num_rows() as u64;
    let input_evidence = PartialAggregateInputEvidence {
        source_chunk_fingerprint: partial_aggregate_source_chunk_fingerprint(&batch.chunks),
        source_adjacent_partial: true,
        source_chunk_row_group_count: batch.row_group_count() as u64,
        source_chunk_reads_all_row_groups: batch.chunks.iter().all(|chunk| chunk.reads_all_row_groups),
        source_chunk_known_row_count: sum_optional_u64(batch.chunks.iter().map(|chunk| chunk.known_row_count)),
        source_chunk_estimated_source_read_bytes: batch
            .estimated_source_read_bytes
            .or_else(|| sum_optional_u64(batch.chunks.iter().map(|chunk| chunk.estimated_source_read_bytes))),
        source_chunk_size_estimate_based: batch.chunks.iter().any(|chunk| chunk.size_estimate_based),
        source_adjacent_partition_strategy: "none",
        source_adjacent_partition_count: None,
        source_adjacent_non_empty_partition_count: None,
        source_adjacent_partition_key_count: None,
        source_adjacent_max_partition_fan_in: Some(batch.chunk_count() as u64),
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
        execution.metrics(),
    );
    execution.metrics().add_chunk_operator_time(chunk_operator_timer.elapsed());
    let frame = match chunk_output {
        Ok(frame) => frame,
        Err(error) => {
            return Err(attach_chunk_resource_facts(error, first_chunk, "chunk_operator", 1, 1, execution.metrics()));
        },
    };

    let partial_bytes = observe_device_memory_bytes(&frame, execution.metrics())
        .map_err(|error| attach_chunk_resource_facts(error, first_chunk, "partial_state", 1, 1, execution.metrics()))?;
    execution.record_source_read(SourceReadFact::ChunkCommitted {
        chunk_count: batch.chunk_count() as u64,
        row_groups: batch.row_group_count() as u64,
    });
    execution.metrics().record_partial_aggregate(partial_input_rows, frame.num_rows() as u64, partial_bytes);

    Ok(CommittedPartial::with_device_bytes(frame, partial_bytes))
}

pub(in crate::exec::bounded::chunks) fn execute_source_adjacent_chunk_stages(
    execution: &mut BoundedExecutionAttempt<'_>,
    bounded: &BoundedPlan<'_>,
    chunk: &SourceChunk,
    mut input: CommittedPartial,
    attempt: usize,
    max_attempts: usize,
) -> Result<CommittedPartial> {
    for (projection, output_schema) in &bounded.pre_filter_projections {
        debug_assert_eq!(projection.output_schema().as_ref(), output_schema.as_ref());
        input = execute_source_adjacent_chunk_projection(
            execution,
            "source_chunk_projection",
            "source_chunk_projection_output",
            chunk,
            projection,
            input,
            attempt,
            max_attempts,
        )?;
    }
    if let Some(filter) = bounded.filter {
        input = execute_source_adjacent_chunk_unary(
            execution,
            "source_chunk_filter",
            "source_chunk_filter_output",
            chunk,
            input,
            attempt,
            max_attempts,
            |context, input, token, metrics| execute_filter_with_budget(context, filter, input, token, metrics),
        )?;
    }
    for (projection, output_schema) in &bounded.post_filter_projections {
        debug_assert_eq!(projection.output_schema().as_ref(), output_schema.as_ref());
        input = execute_source_adjacent_chunk_projection(
            execution,
            "source_chunk_projection",
            "source_chunk_projection_output",
            chunk,
            projection,
            input,
            attempt,
            max_attempts,
        )?;
    }
    Ok(input)
}

#[expect(
    clippy::too_many_arguments,
    reason = "bounded execution passes operator-occurrence inputs explicitly beside the attempt state"
)]
pub(in crate::exec::bounded) fn execute_source_adjacent_chunk_projection(
    execution: &mut BoundedExecutionAttempt<'_>,
    stage: &'static str,
    output_boundary: &'static str,
    chunk: &SourceChunk,
    projection: &ProjectionNode,
    input: CommittedPartial,
    attempt: usize,
    max_attempts: usize,
) -> Result<CommittedPartial> {
    if !projection.outputs.iter().any(|output| output.expr.contains_regexp_replace()) {
        return execute_source_adjacent_chunk_unary(
            execution,
            stage,
            output_boundary,
            chunk,
            input,
            attempt,
            max_attempts,
            |context, input, token, metrics| execute_projection_with_budget(context, projection, input, token, metrics),
        );
    }

    let input_frame = input.frame;
    let prepared =
        match projection_regexp_replace_prepare(execution.context(), projection, &input_frame, execution.metrics()) {
            Ok(Some(prepared)) => prepared,
            Ok(None) => {
                let token = match execution_budget_token_for_boundary(
                    execution.context(),
                    execution.device_budget_bytes(),
                    stage,
                    "nexus_query_engine.source_chunk_unary.budget_token",
                    "failed to create bounded native source-adjacent chunk operator budget token",
                ) {
                    Ok(token) => token,
                    Err(error) => {
                        return Err(attach_chunk_resource_facts(
                            error,
                            chunk,
                            stage,
                            attempt,
                            max_attempts,
                            execution.metrics(),
                        ));
                    },
                };
                let result = execute_projection_with_budget(
                    execution.context(),
                    projection,
                    input_frame,
                    &token,
                    execution.metrics(),
                );
                return finish_source_adjacent_chunk_unary(
                    execution,
                    stage,
                    output_boundary,
                    chunk,
                    result,
                    attempt,
                    max_attempts,
                );
            },
            Err(error) => {
                return Err(attach_chunk_resource_facts(
                    error,
                    chunk,
                    stage,
                    attempt,
                    max_attempts,
                    execution.metrics(),
                ));
            },
        };

    let token = match execution_budget_token_for_boundary(
        execution.context(),
        execution.device_budget_bytes(),
        stage,
        "nexus_query_engine.source_chunk_unary.budget_token",
        "failed to create bounded native source-adjacent chunk operator budget token after prepare reconcile",
    ) {
        Ok(token) => token,
        Err(error) => {
            return Err(attach_chunk_resource_facts(error, chunk, stage, attempt, max_attempts, execution.metrics()));
        },
    };

    let result = execute_projection_with_budget_and_prepared(
        execution.context(),
        projection,
        input_frame,
        &token,
        Some(prepared),
        execution.metrics(),
    );
    finish_source_adjacent_chunk_unary(execution, stage, output_boundary, chunk, result, attempt, max_attempts)
}

fn finish_source_adjacent_chunk_unary(
    execution: &mut BoundedExecutionAttempt<'_>,
    stage: &'static str,
    output_boundary: &'static str,
    chunk: &SourceChunk,
    result: Result<GpuDataFrame>,
    attempt: usize,
    max_attempts: usize,
) -> Result<CommittedPartial> {
    let frame = match result {
        Ok(frame) => frame,
        Err(error) => {
            return Err(attach_chunk_resource_facts(error, chunk, stage, attempt, max_attempts, execution.metrics()));
        },
    };
    let output_bytes = match observe_device_memory_bytes(&frame, execution.metrics()) {
        Ok(bytes) => bytes,
        Err(error) => {
            return Err(attach_chunk_resource_facts(
                error,
                chunk,
                output_boundary,
                attempt,
                max_attempts,
                execution.metrics(),
            ));
        },
    };
    Ok(CommittedPartial::with_device_bytes(frame, output_bytes))
}

#[expect(
    clippy::too_many_arguments,
    reason = "bounded execution passes operator-occurrence inputs explicitly beside the attempt state"
)]
pub(in crate::exec::bounded) fn execute_source_adjacent_chunk_unary(
    execution: &mut BoundedExecutionAttempt<'_>,
    stage: &'static str,
    output_boundary: &'static str,
    chunk: &SourceChunk,
    intermediate_input: CommittedPartial,
    attempt: usize,
    max_attempts: usize,
    execute: impl FnOnce(&ExecutionContext, GpuDataFrame, &ExecutionBudgetToken, &ExecutionMetrics) -> Result<GpuDataFrame>,
) -> Result<CommittedPartial> {
    let token = match execution_budget_token_for_boundary(
        execution.context(),
        execution.device_budget_bytes(),
        stage,
        "nexus_query_engine.source_chunk_unary.budget_token",
        "failed to create bounded native source-adjacent chunk operator budget token",
    ) {
        Ok(token) => token,
        Err(error) => {
            return Err(attach_chunk_resource_facts(error, chunk, stage, attempt, max_attempts, execution.metrics()));
        },
    };
    let result = execute(execution.context(), intermediate_input.frame, &token, execution.metrics());
    let frame = match result {
        Ok(frame) => frame,
        Err(error) => {
            return Err(attach_chunk_resource_facts(error, chunk, stage, attempt, max_attempts, execution.metrics()));
        },
    };
    let output_bytes = match observe_device_memory_bytes(&frame, execution.metrics()) {
        Ok(bytes) => bytes,
        Err(error) => {
            return Err(attach_chunk_resource_facts(
                error,
                chunk,
                output_boundary,
                attempt,
                max_attempts,
                execution.metrics(),
            ));
        },
    };
    Ok(CommittedPartial::with_device_bytes(frame, output_bytes))
}

#[cfg(test)]
mod tests {
    use std::{path::PathBuf, sync::Arc};

    use arrow::datatypes::{DataType, Field, Schema};

    use super::*;
    use crate::{
        runtime::SourceChunkPlanningMode,
        source::{CredentialMode, LocalParquetFile, ParquetSourceLocation, ResolvedParquetSource},
    };

    #[test]
    fn remote_direct_grouped_source_adjacent_batching_uses_grouped_aggregate_cap() {
        let source = remote_direct_source();
        let chunks = vec![remote_direct_chunk(0, 128 * 1024 * 1024), remote_direct_chunk(1, 128 * 1024 * 1024)];

        assert_eq!(
            source_adjacent_raw_source_batch_max_bytes(&source, &chunks, true),
            BoundedSourceBatchingPolicy::DEFAULT_SOURCE_ADJACENT_GROUPED_AGGREGATE_TARGET_CHUNK_BYTES
        );
        assert_eq!(
            source_adjacent_raw_source_batch_max_bytes(&source, &chunks, false),
            BoundedSourceBatchingPolicy::DEFAULT_REMOTE_DIRECT_SOURCE_ADJACENT_RAW_SOURCE_BATCH_MAX_BYTES
        );
    }

    fn remote_direct_source() -> LocalParquetSource {
        let schema = Arc::new(Schema::new(vec![Field::new("c0", DataType::Int64, false)]));
        let file = LocalParquetFile::from_resolved_parquet_source(remote_direct_resolved_source()).unwrap();
        LocalParquetSource::from_files(vec![file], schema, vec!["c0".to_string()])
    }

    fn remote_direct_chunk(chunk_index: usize, estimated_source_read_bytes: u64) -> SourceChunk {
        let resolved_source = remote_direct_resolved_source();
        SourceChunk {
            source_node_id: crate::plan::NodeId::from_index_for_internal_use(0),
            chunk_index,
            file_index: 0,
            location: ParquetSourceLocation::object_store_uri("s3://bucket/table.parquet").unwrap(),
            path: PathBuf::from("s3://bucket/table.parquet"),
            resolved_source,
            stable_id: "s3://bucket/table.parquet".to_string(),
            row_group_indexes: vec![chunk_index],
            reads_all_row_groups: false,
            known_file_bytes: Some(estimated_source_read_bytes),
            known_row_group_bytes: Some(estimated_source_read_bytes),
            known_projected_compressed_bytes: Some(estimated_source_read_bytes),
            known_projected_uncompressed_bytes: Some(estimated_source_read_bytes),
            known_pure_filter_compressed_bytes: Some(0),
            known_pure_filter_uncompressed_bytes: Some(0),
            estimated_chunk_bytes: Some(estimated_source_read_bytes),
            estimated_source_read_bytes: Some(estimated_source_read_bytes),
            exact_source_read_preflight_bytes: None,
            source_file_row_groups: Some(vec![chunk_index]),
            source_file_row_group_bytes: Some(vec![estimated_source_read_bytes]),
            source_file_projected_compressed_bytes: Some(vec![estimated_source_read_bytes]),
            source_file_projected_uncompressed_bytes: Some(vec![estimated_source_read_bytes]),
            source_file_pure_filter_compressed_bytes: Some(vec![0]),
            source_file_pure_filter_uncompressed_bytes: Some(vec![0]),
            source_file_row_group_rows: Some(vec![1]),
            reader_decimal_types: Vec::new(),
            reader_decimal_types_footer_verified: true,
            read_columns: vec!["c0".to_string()],
            partition_values: Vec::new(),
            zero_column_count: false,
            known_row_count: Some(1),
            scan_limit_rows: None,
            size_estimate_based: false,
            source_batching_reason: None,
            source_chunk_planning_mode: SourceChunkPlanningMode::Conservative,
            source_chunk_planning_reason: "test".to_string(),
        }
    }

    fn remote_direct_resolved_source() -> ResolvedParquetSource {
        ResolvedParquetSource::remote_kvikio("s3://bucket/table.parquet", CredentialMode::SharedProcessEnv)
    }
}
