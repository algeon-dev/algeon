//! Bounded aggregates of the compiled pipeline: partial states per input chunk, merged into the
//! final result.
//!
//! A native aggregate task consumes its input as frames or source chunks. It computes one
//! partial-state frame per chunk, then a final merge regroups the partials by group key into the
//! aggregate's output (`website/docs/design/gpu-execution.mdx` describes the chunked execution).
//! `stream.rs` owns the partial stage and the accumulators, `final_merge` the merge,
//! `partitioned.rs` and `source_adjacent.rs` the group-key partitioned variants, `sizing.rs` the
//! partial-state size bound, and `post_stages.rs` the projection after the final aggregate.
//!
//! A `CommittedPartial` (an alias of `ResidentGpuFrame`) is the partial-state frame of one input
//! chunk with its observed device bytes. A chunk's outputs stay private to the chunk while its
//! decode and operator retries run (`AggregateChunkOutput`). The coordinator then inserts the
//! committed partials into merge state, and a merge failure after that point cannot replay the
//! chunk. The accumulator owns a partial from insertion until a merge consumes it. The merged
//! output is itself a `CommittedPartial`, and the inputs are released once it exists.
//!
//! Partial state differs from the output shape. AVG keeps a sum and a count, a global SUM, MIN or
//! MAX keeps a valid count, and COUNT merges by summing. The merge preserves that state, and
//! `post_stages.rs` or `finalize_compiled_final_aggregate_output` divides and drops it.
//!
//! `CompiledStreamPartials::new` picks the accumulator. When the aggregate capability decision
//! selects the partitioned merge for the final stage and the aggregate has no grouping sets, the
//! incremental partitioned merge (`final_merge/incremental.rs`) folds partials into per-partition
//! states while chunks arrive; retained bytes then follow the merged state.
//! Otherwise the partials are retained, or folded in a binary carry, and
//! `execute_compiled_bounded_final_aggregate_merge` (`final_merge/compiled.rs`) runs at the end. It
//! partitions by group key when the same capability decision and the partial sizes allow it, and
//! merges level by level otherwise.
//!
//! No step here reserves memory. The attempt's device budget is a batching input that sizes
//! chunks, partitions and fan-in, and the query allocator enforces the grant when a step
//! allocates. Admission terms (attempt, grant) are defined in
//! `website/docs/design/admission.mdx#from-capability-validation-to-admission`.

use std::sync::Arc;

use arrow::{
    array::new_empty_array,
    datatypes::{DataType as ArrowDataType, SchemaRef as ArrowSchemaRef},
    record_batch::RecordBatchOptions,
};
use query_planning::pipeline::AggregateInputHints;
use query_runtime::observability::{
    RuntimeObservationRecordClass, RuntimeOperatorStageGuard, RuntimeOperatorStageIdentity, RuntimeOperatorStageKind,
    RuntimePlanNodeId, RuntimeTaskId, RuntimeTopologyRecord, TaskDefinitionRow, stage_context_from_execution,
};

use super::{
    AggregateFunction, AggregateNode, AggregateSpec, AvgStateAliases, BoundedExecutionAttempt,
    BoundedJoinLeafLocalParquetSourceStream, BoundedPartitionRowPolicy, BoundedPartitionTargetPolicy,
    BoundedSourceBatchingPolicy, CommittedPartial, DataFrame, ExecutionBudgetToken, ExecutionContext, ExecutionMetrics,
    Expr, Field, FilterNode, GpuDataFrame, Instant, LimitNode, LocalParquetSource, MemoryControlPolicy, NodeId,
    PartialAggregateInputEvidence, PlanningPolicy, ProjectionNode, QueryError, QueryErrorCode, RawSourceInput,
    RecordBatch, ResidentGpuFrame, Result, RuntimeResourceFacts, SortNode, SourceChunk, SourceChunkBatch,
    SourceChunkPlanningPolicy, aggregate_spec_has_avg, aggregate_spec_has_distinct,
    aggregate_spec_has_floating_statistical_aggregate, aggregate_spec_has_median, aggregate_spec_requires_whole_input,
    attach_budget_token_failure_facts, bounded_invariant_violation, budgeted_value,
    cast_count_output_columns_with_budget, compiled::ResidentFrameStreamPayload, cudf_query_error,
    execute_aggregate_with_budget, execute_aggregate_with_budget_for_node, execute_bounded_raw_source_chunk,
    execute_bounded_raw_source_chunk_batch, execute_bounded_union, execute_datafusion_exact_host_aggregate_batch,
    execute_final_sort_node_with_retry, execute_global_aggregate_with_budget,
    execute_groupby_aggregate_with_budget_observed, execute_partial_aggregate_with_budget,
    execution_budget_token_for_boundary, final_global_aggregate_state_output_schema,
    finalize_avg_output_columns_with_budget, global_valid_count_state_alias, is_retryable_chunk_error,
    lower_aggregate_spec_keys, lower_column_expr, lower_final_aggregate_spec, lower_named_column_expr,
    materialize_datafusion_exact_host_aggregate_input, observe_device_memory_bytes,
    partial_aggregate_source_chunk_fingerprint, partial_global_aggregate_output_schema,
    plan_source_chunk_batches_with_row_cap, prepare_aggregate_input_frame,
    prepare_datafusion_exact_host_aggregate_input_frame, record_batch_memory_size, record_chunk_retry_error,
    record_default_grouped_aggregate_strategy_decision, record_device_memory_observation,
    record_grouped_aggregate_strategy_decision, run_budgeted,
    should_execute_datafusion_exact_floating_statistical_aggregate,
    stream_bounded_join_leaf_local_parquet_source_chunks, topk_candidate_rows,
    upload_datafusion_exact_host_aggregate_output, validate_aggregate_output_names,
};
use crate::exec::operators::final_global_aggregate_spec;

mod final_merge;
mod final_merge_policy;
mod partitioned;
mod plan;
mod post_stages;
pub(crate) mod sizing;
mod source_adjacent;
mod stream;

use final_merge::{
    CompiledIncrementalPartitionedMerge, execute_incremental_source_adjacent_final_aggregate_merge,
    should_execute_incremental_partitioned_final_aggregate_merge,
};
pub(super) use final_merge::{execute_compiled_bounded_final_aggregate_merge, reduce_merge_fan_in};
use partitioned::{
    execute_compiled_bounded_partitioned_aggregate_from_frame, partitioned_aggregate_partition_count,
    should_partition_compiled_grouped_aggregate,
};
pub(super) use plan::{BoundedPlan, BoundedUnaryStage, SourceAggregateTopK};
use post_stages::{aggregate_spec_has_global_final_state, execute_final_aggregate_projection};
use sizing::partial_aggregate_work_reservation_upper_bound_bytes;
pub(in crate::exec::bounded) use source_adjacent::execute_source_aggregate_with_dispatch;
pub(in crate::exec::bounded) use stream::{
    execute_compiled_bounded_aggregate_from_source_stream, execute_compiled_bounded_aggregate_from_stream,
    execute_compiled_source_aggregate_chunk,
};

use crate::materialization_sizing::{PartialAggregateShape, partial_aggregate_upper_bound_bytes};

/// Caps the row groups per source chunk for grouped aggregates over a string key when stages sit
/// between the source and the aggregate. The byte target derived from the grant is left unchanged,
/// and the cap is recorded as the planning reason
/// `source_adjacent_grouped_aggregate_downstream_cap`.
pub(in crate::exec::bounded) fn apply_source_adjacent_aggregate_chunk_caps(
    mut policy: SourceChunkPlanningPolicy,
    hints: AggregateInputHints,
) -> SourceChunkPlanningPolicy {
    let batching_policy = BoundedSourceBatchingPolicy::default();
    if hints.grouped_string_key {
        policy.max_row_groups_per_chunk = policy
            .max_row_groups_per_chunk
            .min(batching_policy.source_adjacent_grouped_aggregate_max_operator_row_groups());
        policy.mark_downstream_cap("source_adjacent_grouped_aggregate_downstream_cap");
    }
    policy
}

pub(in crate::exec::bounded) fn is_source_adjacent_grouped_aggregate(bounded: &BoundedPlan<'_>) -> bool {
    bounded.has_source_adjacent_stages()
        && !bounded.aggregate.0.spec.group_exprs.is_empty()
        && bounded.aggregate.0.spec.grouping_sets.is_none()
}

/// Whether a multi-chunk grouped aggregate with source-adjacent stages runs with group-key
/// partitioning (`source_adjacent.rs`).
///
/// Floating statistical aggregates, medians and grouping sets are excluded. A false result selects
/// another path and never fails the query.
pub(in crate::exec::bounded) fn should_execute_partitioned_source_adjacent_grouped_aggregate(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    chunks: &[SourceChunk],
) -> bool {
    // Admission uses the max live incremental working set; whole-input retention and
    // fixed chunk or row-group counts are not admission inputs. Source-adjacent group-key
    // partition is the high-cardinality path for multi-chunk string grouped aggregates.
    bounded.has_source_adjacent_stages()
        && !bounded.aggregate.0.spec.group_exprs.is_empty()
        && !aggregate_spec_has_floating_statistical_aggregate(&bounded.aggregate.0.spec)
        && !aggregate_spec_has_median(&bounded.aggregate.0.spec)
        && bounded.aggregate.0.spec.grouping_sets.is_none()
        && chunks.len() > 1
        && source_adjacent_partitioned_live_working_set_fits_budget(context, bounded, chunks)
}

/// Estimates the live working set of the partitioned path and compares it with the attempt's device
/// budget, which is a batching input and not a reservation. An unknown chunk size decides by key
/// type: only a string-keyed aggregate proceeds, and other shapes stay on the capability-first
/// path.
fn source_adjacent_partitioned_live_working_set_fits_budget(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    chunks: &[SourceChunk],
) -> bool {
    let MemoryControlPolicy { device_budget_bytes, .. } = context.memory_control_policy();

    // The live set is approximated as the sum of these terms:
    //   one source batch + projected/hash-partition peak + one ready partition batch
    //   + retained partition-local partial floors across partitions.
    let mut largest_source_batch_bytes = 0u64;
    let mut known_chunk_bytes = 0u64;
    let mut unknown_chunk_bytes = false;
    for chunk in chunks {
        match chunk.planning_bytes() {
            Some(chunk_bytes) => {
                largest_source_batch_bytes = largest_source_batch_bytes.max(chunk_bytes);
                known_chunk_bytes = known_chunk_bytes.saturating_add(chunk_bytes);
            },
            None => unknown_chunk_bytes = true,
        }
    }

    // String-key multi-chunk aggregates may stream under group-key partitioning even
    // when some chunk sizes are unknown; non-string shapes remain capability-first.
    if unknown_chunk_bytes {
        return source_adjacent_grouped_aggregate_has_string_key(bounded);
    }
    if largest_source_batch_bytes == 0 {
        return true;
    }

    let partition_count = partitioned_aggregate_partition_count(context, known_chunk_bytes.max(1)).max(
        BoundedPartitionRowPolicy::default().source_adjacent_partitioned_aggregate_min_partition_count(chunks.len()),
    );

    let batching = BoundedSourceBatchingPolicy::default();
    let ready_partition_batch_bytes =
        batching.source_adjacent_partition_batch_byte_limit_for_budget(device_budget_bytes);
    // Projected + hash-partition peak roughly tracks the live source batch size twice
    // (input retained while output is built) before the input is released.
    let projected_and_hash_bytes = largest_source_batch_bytes.saturating_mul(2);
    // Floor retained partial state without assuming full raw retention.
    let retained_partial_floor = (largest_source_batch_bytes / (partition_count as u64).max(1)).max(1);
    let retained_partial_cap = device_budget_bytes / 4;
    let live_working_set = largest_source_batch_bytes
        .saturating_add(projected_and_hash_bytes)
        .saturating_add(ready_partition_batch_bytes)
        .saturating_add(retained_partial_floor.saturating_mul(partition_count as u64).min(retained_partial_cap));

    live_working_set <= device_budget_bytes
}

fn source_adjacent_grouped_aggregate_has_string_key(bounded: &BoundedPlan<'_>) -> bool {
    let Ok(lowered_keys) = lower_aggregate_spec_keys(&bounded.aggregate.0.spec) else {
        return false;
    };
    lowered_keys.key_columns.iter().any(|key_column| {
        bounded.aggregate_input_schema.field_with_name(key_column).is_ok_and(|field| {
            matches!(field.data_type(), ArrowDataType::Utf8 | ArrowDataType::LargeUtf8 | ArrowDataType::Utf8View)
        })
    })
}

/// Single-stage aggregate over one resident frame.
///
/// A floating statistical aggregate in DataFusion-exact mode runs on the host
/// (`execute_compiled_bounded_exact_host_aggregate_from_frame`). A grouped aggregate whose input
/// exceeds the aggregate partition byte target is hash-partitioned on its group keys, aggregated
/// per partition and merged (`partitioned.rs`). Anything else runs once under a budget token for
/// `compiled_aggregate`.
pub(in crate::exec::bounded) fn execute_compiled_bounded_aggregate_from_frame(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &ArrowSchemaRef,
    output_schema: &ArrowSchemaRef,
    input: GpuDataFrame,
    input_device_bytes: u64,
) -> Result<ResidentGpuFrame> {
    if should_execute_datafusion_exact_floating_statistical_aggregate(execution.context(), &aggregate.spec) {
        return execute_compiled_bounded_exact_host_aggregate_from_frame(
            execution,
            node_id,
            aggregate,
            input_schema,
            output_schema,
            input,
        );
    }
    if should_partition_compiled_grouped_aggregate(execution.context(), aggregate, input_device_bytes) {
        return execute_compiled_bounded_partitioned_aggregate_from_frame(
            execution,
            node_id,
            aggregate,
            input_schema,
            output_schema,
            input,
            input_device_bytes,
        );
    }

    run_budgeted(execution, node_id, "compiled_aggregate", input, |context, input, token, metrics| {
        execute_aggregate_with_budget_for_node(
            context,
            Some(node_id),
            aggregate,
            input_schema,
            input,
            output_schema,
            token,
            metrics,
        )
    })
}

pub(in crate::exec::bounded) fn execute_compiled_bounded_aggregate_consuming_frame(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &ArrowSchemaRef,
    output_schema: &ArrowSchemaRef,
    input: ResidentGpuFrame,
) -> Result<ResidentGpuFrame> {
    let input_device_bytes = input.device_bytes();
    execute_compiled_bounded_aggregate_from_frame(
        execution,
        node_id,
        aggregate,
        input_schema,
        output_schema,
        input.frame,
        input_device_bytes,
    )
}

/// DataFusion-exact floating statistical aggregate. The prepared device input is copied to a host
/// Arrow batch, accumulated on the host in DataFusion's order, and the result is uploaded again.
/// The device input is consumed by the copy.
fn execute_compiled_bounded_exact_host_aggregate_from_frame(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &ArrowSchemaRef,
    output_schema: &ArrowSchemaRef,
    input: GpuDataFrame,
) -> Result<ResidentGpuFrame> {
    let stage = "compiled_exact_host_welford_aggregate";
    let token = match execution_budget_token_for_boundary(
        execution.context(),
        execution.device_budget_bytes(),
        stage,
        "query_engine.bounded.aggregate.exact_host_welford.budget_token",
        "failed to create bounded exact host-Welford aggregate budget token",
    ) {
        Ok(token) => token,
        Err(error) => {
            return Err(error.with_fact_if_absent("node_id", node_id));
        },
    };

    let (prepared, spec, prepared_input_schema) = match prepare_aggregate_input_frame(
        execution.context(),
        input.into_frame(),
        input_schema,
        &aggregate.spec,
        Some(&token),
        execution.metrics(),
    ) {
        Ok(prepared) => prepared,
        Err(error) => {
            return Err(error.with_fact_if_absent("node_id", node_id));
        },
    };
    let prepared = GpuDataFrame::new(prepared);
    let (exact_input, exact_input_schema) = match prepare_datafusion_exact_host_aggregate_input_frame(
        execution.context(),
        prepared,
        &prepared_input_schema,
        execution.metrics(),
    ) {
        Ok(input) => input,
        Err(error) => {
            return Err(error.with_fact_if_absent("node_id", node_id));
        },
    };
    observe_device_memory_bytes(&exact_input, execution.metrics())
        .map_err(|error| error.with_fact_if_absent("node_id", node_id))?;
    let batch =
        match materialize_datafusion_exact_host_aggregate_input(execution.context(), exact_input, exact_input_schema) {
            Ok(batch) => batch,
            Err(error) => {
                return Err(error.with_fact_if_absent("node_id", node_id));
            },
        };

    let output_batch = execute_datafusion_exact_host_aggregate_batch(
        execution.context(),
        Some(node_id),
        &spec,
        &batch,
        output_schema,
        execution.metrics(),
        None,
    )?;
    let _ = record_batch_memory_size(&output_batch);
    let frame = match upload_datafusion_exact_host_aggregate_output(execution.context(), &output_batch)
        .map(GpuDataFrame::new)
    {
        Ok(frame) => frame,
        Err(error) => {
            return Err(error.with_fact_if_absent("node_id", node_id));
        },
    };
    let output_bytes = match observe_device_memory_bytes(&frame, execution.metrics()) {
        Ok(bytes) => bytes,
        Err(error) => {
            return Err(error.with_fact_if_absent("node_id", node_id));
        },
    };
    Ok(ResidentGpuFrame::with_device_bytes(frame, output_bytes))
}

/// Aggregates zero rows: a zero-row frame in the input schema goes through the single-stage path,
/// so a global aggregate returns its identity row and grouping sets their grand-total row.
pub(in crate::exec::bounded::aggregate) fn execute_compiled_bounded_aggregate_from_empty_input(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &ArrowSchemaRef,
    output_schema: &ArrowSchemaRef,
) -> Result<ResidentGpuFrame> {
    let input = empty_resident_frame_for_schema(execution, node_id, input_schema)?;
    let input_device_bytes = input.device_bytes();
    execute_compiled_bounded_aggregate_from_frame(
        execution,
        node_id,
        aggregate,
        input_schema,
        output_schema,
        input.frame,
        input_device_bytes,
    )
}

fn empty_resident_frame_for_schema(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    schema: &ArrowSchemaRef,
) -> Result<ResidentGpuFrame> {
    let arrays = schema.fields().iter().map(|field| new_empty_array(field.data_type())).collect::<Vec<_>>();
    let batch = RecordBatch::try_new_with_options(
        Arc::clone(schema),
        arrays,
        &RecordBatchOptions::new().with_row_count(Some(0)),
    )
    .map_err(|error| {
        QueryError::new(
            QueryErrorCode::Internal,
            "query_engine.bounded.aggregate.empty_input",
            "failed to construct zero-row aggregate input batch",
        )
        .with_fact("node_id", node_id)
        .with_source(error)
    })?;
    let frame = DataFrame::from_record_batch(&execution.context().runtime()?, &batch).map_err(|error| {
        cudf_query_error(
            "query_engine.bounded.aggregate.empty_input",
            "failed to upload zero-row aggregate input",
            error,
        )
        .with_fact("node_id", node_id)
    })?;
    let frame = GpuDataFrame::new(frame);
    let output_bytes = observe_device_memory_bytes(&frame, execution.metrics())?;
    Ok(ResidentGpuFrame::with_device_bytes(frame, output_bytes))
}

/// Correlates source-owned partial kernels with the chunk attempt and lane
/// whose stream executes them, without adding a second observation schema.
pub(in crate::exec::bounded) fn start_aggregate_chunk_stage(
    execution: &BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    chunk_index: usize,
    attempt: usize,
    input_rows: u64,
) -> RuntimeOperatorStageGuard {
    let (observation, plan_execution_id, parent_span_id) = stage_context_from_execution(execution.context());
    let fork =
        execution.context().attempt_claimed_fork_index().map_or_else(|| "root".to_string(), |fork| fork.to_string());
    let task_id = RuntimeTaskId::new(format!("aggregate.{node_id}.chunk{chunk_index}.fork{fork}",));
    if let Some(plan_execution_id) = &plan_execution_id {
        observation.record_topology(RuntimeObservationRecordClass::TaskDefinition, || {
            RuntimeTopologyRecord::TaskDefinition(TaskDefinitionRow {
                plan_execution_id: plan_execution_id.clone(),
                task_id: task_id.clone(),
                plan_node_id: Some(RuntimePlanNodeId::new(node_id.to_string())),
                task_kind: "aggregate_chunk".to_string(),
            })
        });
    }
    RuntimeOperatorStageGuard::start(
        observation,
        plan_execution_id,
        parent_span_id,
        RuntimeOperatorStageKind::PartialAggregate,
        RuntimeOperatorStageIdentity::new()
            .with_node_id(node_id.to_string())
            .with_task_id(task_id)
            .with_chunk_index(chunk_index)
            .with_attempt(attempt as u32)
            .with_input_rows(input_rows),
    )
}

#[cfg(test)]
mod tests {
    use query_planning::execution_contracts::SourceChunkPlanningMode;

    use super::*;

    fn throughput_like_policy() -> SourceChunkPlanningPolicy {
        let mut policy = SourceChunkPlanningPolicy::new(Some(32 * 1024 * 1024 * 1024), 1024);
        // Mirror ParquetThroughput widening without needing a real source.
        policy.max_row_groups_per_chunk = 1024;
        policy.mark_downstream_cap("parquet_throughput_admitted");
        let _ = SourceChunkPlanningMode::ParquetThroughput;
        policy
    }

    #[test]
    fn string_key_row_group_cap_preserves_grant_derived_byte_target() {
        let policy = apply_source_adjacent_aggregate_chunk_caps(
            throughput_like_policy(),
            AggregateInputHints { adjacent_stages: true, grouped_string_key: true },
        );

        assert_eq!(
            policy.max_row_groups_per_chunk,
            BoundedSourceBatchingPolicy::DEFAULT_SOURCE_ADJACENT_GROUPED_AGGREGATE_MAX_OPERATOR_ROW_GROUPS
        );
        assert_eq!(policy.max_source_chunk_bytes, Some(32 * 1024 * 1024 * 1024));
        assert_eq!(policy.source_chunk_planning_reason(), "source_adjacent_grouped_aggregate_downstream_cap");
    }
}
