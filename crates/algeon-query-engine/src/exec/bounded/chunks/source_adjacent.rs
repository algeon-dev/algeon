use std::time::Instant;

use super::{
    BoundedExecutionAttempt, BoundedPlan, BoundedSourceBatchingPolicy, CommittedPartial, ExecutionBudgetToken,
    ExecutionContext, ExecutionMetrics, GpuDataFrame, LocalParquetSource, ParquetReadStrategy,
    PartialAggregateInputEvidence, ProjectionNode, Result, ScheduledLocalParquetReadOutput, SourceChunk,
    SourceChunkBatch, attach_chunk_resource_facts, execute_filter_with_budget, execute_partial_aggregate_with_budget,
    execute_projection_with_budget, execute_projection_with_budget_and_prepared, execution_budget_token_for_boundary,
    is_retryable_chunk_error, is_source_adjacent_grouped_aggregate, observe_device_memory_bytes,
    partial_aggregate_source_chunk_fingerprint, plan_source_chunk_batches, projection_regexp_replace_prepare,
    read_local_parquet_chunk_batch_required_only_no_commit, record_chunk_retry_error,
};
use crate::{
    exec::retry::backoff_before_retry, observability::SourceReadFact, rapids_backend::rapids_analysis_context,
    util::sum_optional_u64,
};

pub(in crate::exec::bounded) fn bounded_source_adjacent_chunk_batching_enabled(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    chunks: &[SourceChunk],
) -> bool {
    bounded.has_source_adjacent_stages()
        && bounded.source.partition_columns.is_empty()
        && bounded.source.scan_limit.is_none()
        && bounded.source.reader_predicate.as_ref().is_none_or(|predicate| {
            bounded.source.reader_predicate_is_supported(predicate, rapids_analysis_context(context.planning_policy()))
        })
        && bounded_source_adjacent_chunk_batches(context, bounded, chunks).iter().any(|batch| batch.chunk_count() > 1)
}

pub(in crate::exec::bounded) fn bounded_source_adjacent_chunk_batches(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    chunks: &[SourceChunk],
) -> Vec<SourceChunkBatch> {
    bounded_source_adjacent_chunk_batches_for_work_width(context, bounded, chunks, 1)
}

pub(in crate::exec::bounded) fn bounded_source_adjacent_chunk_batches_for_work_width(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    chunks: &[SourceChunk],
    work_width: usize,
) -> Vec<SourceChunkBatch> {
    let batching_policy = BoundedSourceBatchingPolicy::default();
    let max_batch_source_read_bytes = batching_policy.source_adjacent_raw_source_batch_target_bytes(
        context.memory_control_policy(),
        source_adjacent_source_requires_remote_direct_read(bounded.source, chunks),
        is_source_adjacent_grouped_aggregate(bounded),
    );
    let batches = plan_source_chunk_batches(chunks, max_batch_source_read_bytes);
    if work_width <= 1 || batches.len() >= work_width {
        return batches;
    }
    let Some(total_bytes) = batches
        .iter()
        .map(|batch| batch.estimated_source_read_bytes)
        .try_fold(0u64, |total, bytes| bytes.and_then(|bytes| total.checked_add(bytes)))
    else {
        return batches;
    };
    // One large read cannot fill the run's independently leased work lanes.
    // Cap batch size by the same source-byte evidence used by the planner.
    let lane_target_bytes = total_bytes.div_ceil(work_width as u64).max(1);
    plan_source_chunk_batches(chunks, Some(max_batch_source_read_bytes.unwrap_or(u64::MAX).min(lane_target_bytes)))
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

pub(in crate::exec::bounded) struct SourceAdjacentBatchOutput {
    pub(in crate::exec::bounded) partials: Vec<CommittedPartial>,
    pub(in crate::exec::bounded) chunk_count: u64,
    pub(in crate::exec::bounded) row_groups: u64,
}

/// Keeps all retry outputs private until the coordinator commits the batch.
pub(in crate::exec::bounded) fn execute_source_adjacent_batch_uncommitted(
    execution: &mut BoundedExecutionAttempt<'_>,
    bounded: &BoundedPlan<'_>,
    batch: SourceChunkBatch,
) -> Result<SourceAdjacentBatchOutput> {
    execute_source_adjacent_batch_retry(execution, bounded, batch, 1)
}

fn execute_source_adjacent_batch_retry(
    execution: &mut BoundedExecutionAttempt<'_>,
    bounded: &BoundedPlan<'_>,
    batch: SourceChunkBatch,
    attempt: usize,
) -> Result<SourceAdjacentBatchOutput> {
    use super::execution::read_bounded_chunk_source;
    execution.context().check_cancellation("query_engine.exec.bounded.source_adjacent_chunk_batch_retry")?;
    let super::MemoryControlPolicy { max_retry_attempts, min_retry_chunk_row_groups, .. } =
        execution.context().memory_control_policy();
    let max_attempts = max_retry_attempts.max(1);
    let source = if batch.is_singleton() {
        read_bounded_chunk_source(execution, bounded.source, &batch.chunks[0]).map_err(|error| {
            attach_chunk_resource_facts(
                error,
                &batch.chunks[0],
                "source_chunk_read",
                attempt,
                max_attempts,
                execution.metrics(),
            )
        })
    } else {
        read_local_parquet_chunk_batch_required_only_no_commit(execution, bounded.source, &batch, "source_chunk_read")
    };
    let result = source.and_then(|source| {
        execute_source_adjacent_batch_operator_uncommitted(execution, bounded, &batch, source, attempt, max_attempts)
    });
    match result {
        Ok(output) => Ok(output),
        Err(error) if is_retryable_chunk_error(&error) => {
            record_chunk_retry_error(&error, execution.metrics());
            let chunk_count = batch.chunk_count() as u64;
            let error = if batch.is_singleton() {
                super::chunk_retry_facts(&batch.chunks[0], execution).attach_to_error(error)
            } else {
                error
            };
            let children = if batch.is_singleton() {
                if attempt < max_attempts {
                    super::split_retryable_chunk(&batch.chunks[0], min_retry_chunk_row_groups.max(1)).map(
                        |(left, right)| {
                            let child = |chunk| SourceChunkBatch {
                                batch_index: batch.batch_index,
                                chunks: vec![chunk],
                                estimated_source_read_bytes: None,
                                source_batching_reason: "retry_split".to_string(),
                            };
                            (child(left), child(right), attempt + 1)
                        },
                    )
                } else {
                    None
                }
            } else {
                batch.split_for_retry().map(|(left, right)| (left, right, 1))
            };
            let Some((left, right, next_attempt)) = children else {
                return Err(error);
            };
            execution.record_source_read(SourceReadFact::ChunkRetried);
            backoff_before_retry(
                execution.context(),
                "query_engine.exec.bounded.source_adjacent_chunk_batch_split_retry",
            )?;
            let mut output = execute_source_adjacent_batch_retry(execution, bounded, left, next_attempt)?;
            let right = match execute_source_adjacent_batch_retry(execution, bounded, right, next_attempt) {
                Ok(right) => right,
                Err(error) => {
                    return Err(error);
                },
            };
            output.partials.extend(right.partials);
            output.chunk_count = chunk_count;
            output.row_groups = output.row_groups.saturating_add(right.row_groups);
            Ok(output)
        },
        Err(error) => Err(error),
    }
}

fn execute_source_adjacent_batch_operator_uncommitted(
    execution: &mut BoundedExecutionAttempt<'_>,
    bounded: &BoundedPlan<'_>,
    batch: &SourceChunkBatch,
    source: ScheduledLocalParquetReadOutput,
    attempt: usize,
    max_attempts: usize,
) -> Result<SourceAdjacentBatchOutput> {
    let first_chunk = batch.chunks.first().expect("source chunk batch is non-empty");
    let ScheduledLocalParquetReadOutput { frame, observed_bytes, observed_row_group_count, .. } = source;
    let row_groups = super::execution::reported_source_chunk_batch_read_row_groups(batch, observed_row_group_count);
    execution.context().check_cancellation("query_engine.exec.bounded.chunk_batch_operator")?;
    let mut input = CommittedPartial::with_device_bytes(frame, observed_bytes);
    input = execute_source_adjacent_chunk_stages(execution, bounded, first_chunk, input, attempt, max_attempts)?;
    let CommittedPartial { frame, .. } = input;

    let operator_token_budget = execution.device_budget_bytes();
    let operator_token = match execution_budget_token_for_boundary(
        execution.context(),
        operator_token_budget,
        "chunk_operator",
        "query_engine.chunk_operator.budget_token",
        "failed to create bounded native chunk batch operator budget token",
    ) {
        Ok(token) => token,
        Err(error) => {
            return Err(attach_chunk_resource_facts(
                error,
                first_chunk,
                "chunk_operator",
                attempt,
                max_attempts,
                execution.metrics(),
            ));
        },
    };

    let partial_input_rows = frame.num_rows() as u64;
    let input_evidence = PartialAggregateInputEvidence {
        source_chunk_fingerprint: partial_aggregate_source_chunk_fingerprint(&batch.chunks),
        source_adjacent_partial: !batch.is_singleton() || bounded.has_source_adjacent_stages(),
        source_chunk_row_group_count: row_groups,
        source_chunk_reads_all_row_groups: batch.chunks.iter().all(|chunk| chunk.reads_all_row_groups),
        source_chunk_known_row_count: sum_optional_u64(batch.chunks.iter().map(|chunk| chunk.known_row_count)),
        source_chunk_estimated_source_read_bytes: if batch.is_singleton() {
            first_chunk.estimated_source_read_bytes
        } else {
            batch
                .estimated_source_read_bytes
                .or_else(|| sum_optional_u64(batch.chunks.iter().map(|chunk| chunk.estimated_source_read_bytes)))
        },
        source_chunk_size_estimate_based: batch.chunks.iter().any(|chunk| chunk.size_estimate_based),
        source_adjacent_partition_strategy: "none",
        source_adjacent_partition_count: None,
        source_adjacent_non_empty_partition_count: None,
        source_adjacent_partition_key_count: None,
        source_adjacent_max_partition_fan_in: (!batch.is_singleton()).then_some(batch.chunk_count() as u64),
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
            return Err(attach_chunk_resource_facts(
                error,
                first_chunk,
                "chunk_operator",
                attempt,
                max_attempts,
                execution.metrics(),
            ));
        },
    };

    let partial_bytes = observe_device_memory_bytes(&frame, execution.metrics()).map_err(|error| {
        attach_chunk_resource_facts(error, first_chunk, "partial_state", attempt, max_attempts, execution.metrics())
    })?;
    execution.metrics().record_partial_aggregate(partial_input_rows, frame.num_rows() as u64, partial_bytes);

    Ok(SourceAdjacentBatchOutput {
        partials: vec![CommittedPartial::with_device_bytes(frame, partial_bytes)],
        chunk_count: batch.chunk_count() as u64,
        row_groups,
    })
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
                    "query_engine.source_chunk_unary.budget_token",
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
        "query_engine.source_chunk_unary.budget_token",
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
        "query_engine.source_chunk_unary.budget_token",
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
        plan::NodeId,
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
            source_node_id: NodeId::from_index_for_test(0),
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
