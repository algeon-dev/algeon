//! Partial-aggregate stage of the compiled aggregate, fed by a frame stream or a source stream.
//!
//! `execute_compiled_bounded_aggregate_from_stream` takes resident frames, such as a join output.
//! `execute_compiled_bounded_aggregate_from_source_stream` takes a deferred Parquet leaf. Both pick
//! one of these paths from the aggregate spec:
//!
//! - Aggregates that need all rows at once (median, floating statistical aggregates in
//!   DataFusion-exact mode, DISTINCT) union their input and run one single-stage aggregate.
//! - GPU-native statistical aggregates keep one statistical partial per frame and merge them in
//!   one step.
//! - Everything else computes a partial per frame or chunk, with a proactive split when the
//!   partial's size bound exceeds the device budget (`sizing.rs`). The partials enter a
//!   `CompiledStreamPartials` accumulator, which merges them as described in `aggregate.rs`.
//!
//! A source stream whose plan has source-adjacent stages can instead go to `source_adjacent.rs`,
//! which fuses the stages with group-key partitioning.

use arrow::datatypes::SchemaRef as ArrowSchemaRef;

use super::{
    AggregateNode, BoundedExecutionAttempt, BoundedJoinLeafLocalParquetSourceStream, CommittedPartial,
    CompiledIncrementalPartitionedMerge, ExecutionContext, GpuDataFrame, NodeId, ResidentFrameStreamPayload,
    ResidentGpuFrame, Result, SourceChunk, aggregate_spec_has_distinct,
    aggregate_spec_has_floating_statistical_aggregate, aggregate_spec_has_median, aggregate_spec_requires_whole_input,
    attach_budget_token_failure_facts, bounded_invariant_violation, cudf_query_error, execute_aggregate_with_budget,
    execute_bounded_union, execute_compiled_bounded_aggregate_consuming_frame,
    execute_compiled_bounded_aggregate_from_empty_input, execute_compiled_bounded_aggregate_from_frame,
    execute_compiled_bounded_final_aggregate_merge, execute_partial_aggregate_with_budget,
    execution_budget_token_for_boundary,
    final_merge::{execute_compiled_final_merge_group_attempt, normalize_compiled_aggregate_partial},
    observe_device_memory_bytes, partial_aggregate_work_reservation_upper_bound_bytes, run_budgeted,
    should_execute_incremental_partitioned_final_aggregate_merge, stream_bounded_join_leaf_local_parquet_source_chunks,
};
use crate::exec::{
    bounded::{
        chunks::bounded_source_adjacent_chunk_batching_enabled,
        compiled::{CompiledGpuResource, lanes::AggregateLaneDispatch},
        join::bounded_join_leaf_local_parquet_source_frame_stream,
    },
    operators::{
        PartialAggregateInputEvidence, execute_gpu_native_statistical_final_merge_with_budget,
        execute_gpu_native_statistical_partial_aggregate_with_budget, partial_aggregate_source_chunk_fingerprint,
        should_execute_gpu_native_statistical_partial_aggregate,
    },
    retry::RetryLimits,
    source::observed_chunk_row_group_count,
};

/// Aggregates a stream of resident frames into the final output.
///
/// Consumes the stream. The frame stream's producer is released on every path, and frames not yet
/// pulled are released when a partial or a merge step fails. Dispatch tests, in order, for an empty
/// input, GPU-native statistical partials, whole-input aggregates (median and floating statistical
/// aggregates), DISTINCT, and finally the partial and merge path.
pub(in crate::exec::bounded) fn execute_compiled_bounded_aggregate_from_stream(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &ArrowSchemaRef,
    output_schema: &ArrowSchemaRef,
    mut inputs: ResidentFrameStreamPayload,
    preserve_output_stream: bool,
) -> Result<CompiledGpuResource> {
    let result = (|| -> Result<CompiledGpuResource> {
        if inputs.is_empty() {
            return execute_compiled_bounded_aggregate_from_empty_input(
                execution,
                node_id,
                aggregate,
                input_schema,
                output_schema,
            )
            .map(CompiledGpuResource::new);
        }
        if should_execute_gpu_native_statistical_partial_aggregate(execution.context(), &aggregate.spec) {
            return execute_compiled_bounded_statistical_aggregate_from_stream(
                execution,
                node_id,
                aggregate,
                input_schema,
                output_schema,
                &mut inputs,
            )
            .map(CompiledGpuResource::new);
        }
        if aggregate_spec_has_floating_statistical_aggregate(&aggregate.spec)
            || aggregate_spec_has_median(&aggregate.spec)
        {
            let frames = inputs.collect_remaining_frames_owned(execution)?;
            if frames.is_empty() {
                return execute_compiled_bounded_aggregate_from_empty_input(
                    execution,
                    node_id,
                    aggregate,
                    input_schema,
                    output_schema,
                )
                .map(CompiledGpuResource::new);
            }
            let input = execute_bounded_union(execution, node_id, frames)?;
            return execute_compiled_bounded_aggregate_consuming_frame(
                execution,
                node_id,
                aggregate,
                input_schema,
                output_schema,
                input,
            )
            .map(CompiledGpuResource::new);
        }
        if aggregate_spec_has_distinct(&aggregate.spec) {
            let frames = inputs.collect_remaining_frames_owned(execution)?;
            if frames.is_empty() {
                return execute_compiled_bounded_aggregate_from_empty_input(
                    execution,
                    node_id,
                    aggregate,
                    input_schema,
                    output_schema,
                )
                .map(CompiledGpuResource::new);
            }
            return execute_compiled_bounded_distinct_aggregate_from_stream(
                execution,
                node_id,
                aggregate,
                input_schema,
                output_schema,
                frames,
            )
            .map(CompiledGpuResource::new);
        }

        let mut partials = CompiledStreamPartials::new(
            execution,
            node_id,
            aggregate,
            input_schema,
            output_schema,
            inputs.len(),
            Some(inputs.num_rows()),
        )?;
        while let Some(input) = inputs.next_frame(execution) {
            let input = match input {
                Ok(input) => input,
                Err(error) => {
                    return Err(error);
                },
            };
            let chunk_partials = match execute_compiled_bounded_partial_aggregate_from_frame_with_retry(
                execution,
                node_id,
                aggregate,
                input_schema,
                output_schema,
                input,
                None,
            ) {
                Ok(chunk_partials) => chunk_partials,
                Err(error) => {
                    inputs.release_remaining_frames();
                    return Err(error);
                },
            };
            if let Err(error) = partials.extend(execution, chunk_partials) {
                inputs.release_remaining_frames();
                return Err(error);
            }
        }

        partials.finish(execution, preserve_output_stream)
    })();
    inputs.release_producer(execution);
    result
}

#[expect(clippy::too_many_arguments, reason = "compiled aggregate carries its source and borrowed run lanes")]
/// Aggregates a deferred Parquet leaf into the final output.
///
/// A spec that requires the whole input is tested first, which sends every floating statistical
/// aggregate to the union path before the GPU-native statistical check runs, unlike the frame
/// stream path above. Next, a plan with source-adjacent stages goes to `source_adjacent.rs` when it
/// partitions by group key or chunk batching is enabled. The remaining case computes one partial
/// per source chunk, across lanes through `dispatch` when one is given and the spec has no grouping
/// sets, and serially otherwise.
///
/// On the serial path a merge failure ends the aggregate, because source work cannot be replayed
/// after earlier partials were committed. `preserve_output_stream` asks the incremental merge
/// for a lazy frame stream. Without it, the incremental merge concatenates its
/// partitions into one frame.
pub(in crate::exec::bounded) fn execute_compiled_bounded_aggregate_from_source_stream(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &ArrowSchemaRef,
    output_schema: &ArrowSchemaRef,
    stream: BoundedJoinLeafLocalParquetSourceStream,
    preserve_output_stream: bool,
    dispatch: Option<&AggregateLaneDispatch>,
    topk: Option<&super::SourceAggregateTopK<'_>>,
) -> Result<CompiledGpuResource> {
    if aggregate_spec_requires_whole_input(&aggregate.spec) {
        // Whole-input consumers use the same source-owned pull cursor. They
        // must not start nested prefetch workers while the compiled run owns
        // the attempt's lane leases.
        let inputs = bounded_join_leaf_local_parquet_source_frame_stream(execution, stream)
            .into_remaining_frames_owned(execution)?;
        if inputs.is_empty() {
            return execute_compiled_bounded_aggregate_from_empty_input(
                execution,
                node_id,
                aggregate,
                input_schema,
                output_schema,
            )
            .map(CompiledGpuResource::new);
        }
        let input = execute_bounded_union(execution, node_id, inputs)?;
        return execute_compiled_bounded_aggregate_consuming_frame(
            execution,
            node_id,
            aggregate,
            input_schema,
            output_schema,
            input,
        )
        .map(CompiledGpuResource::new);
    }
    if should_execute_gpu_native_statistical_partial_aggregate(execution.context(), &aggregate.spec) {
        return execute_compiled_bounded_statistical_aggregate_from_source_stream(
            execution,
            node_id,
            aggregate,
            input_schema,
            output_schema,
            stream,
        )
        .map(CompiledGpuResource::new);
    }

    if let Some(mut bounded) = stream.aggregate_plan(node_id, aggregate, input_schema, output_schema) {
        if let Some(topk) = topk {
            bounded.post_aggregate_stages = topk.post_aggregate_stages.clone();
            bounded.sort = Some(topk.sort);
            bounded.limit = Some(topk.limit);
        }
        let partitioned = super::should_execute_partitioned_source_adjacent_grouped_aggregate(
            execution.context(),
            &bounded,
            stream.chunks(),
        );
        if partitioned || bounded_source_adjacent_chunk_batching_enabled(execution.context(), &bounded, stream.chunks())
        {
            return super::execute_source_aggregate_with_dispatch(
                execution,
                &bounded,
                stream.chunks(),
                dispatch,
                partitioned,
            );
        }
    }

    let mut partials = CompiledStreamPartials::new(
        execution,
        node_id,
        aggregate,
        input_schema,
        output_schema,
        stream.chunk_count(),
        None,
    )?;
    if let Some(dispatch) = dispatch.filter(|_| aggregate.spec.grouping_sets.is_none()) {
        dispatch.execute_chunks(
            execution,
            node_id,
            aggregate,
            input_schema,
            output_schema,
            stream,
            |execution, chunk_partials| partials.extend(execution, chunk_partials),
        )?;
    } else {
        for chunk in stream.chunks().iter().cloned() {
            let chunk_partials = execute_compiled_source_aggregate_chunk(
                execution,
                node_id,
                aggregate,
                input_schema,
                output_schema,
                &stream,
                chunk,
            )?;
            // Merge failure cannot replay source work after earlier partials committed.
            partials.extend(execution, chunk_partials.partials)?;
            execution.metrics().record_source_chunk_commit(chunk_partials.source_row_groups);
        }
    }
    partials.finish(execution, preserve_output_stream)
}

/// A chunk's outputs stay private through decode/operator retries. Only the
/// coordinator may insert the returned partials into final-merge state.
pub(in crate::exec::bounded) struct AggregateChunkOutput {
    pub(in crate::exec::bounded) partials: Vec<CommittedPartial>,
    pub(in crate::exec::bounded) source_row_groups: u64,
}

/// Computes the committed partials of one source chunk.
///
/// The stream's chunk execution owns decode and operator retries. Each successful piece of the
/// chunk adds its partials to the returned `AggregateChunkOutput`, and the caller inserts them into
/// merge state. A stage record ties each piece's partial aggregate to the chunk index and attempt.
pub(in crate::exec::bounded) fn execute_compiled_source_aggregate_chunk(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &ArrowSchemaRef,
    output_schema: &ArrowSchemaRef,
    stream: &BoundedJoinLeafLocalParquetSourceStream,
    chunk: SourceChunk,
) -> Result<AggregateChunkOutput> {
    let mut partials = Vec::new();
    let source_row_groups = stream.execute_chunk(execution, chunk, &mut |chunk, attempt, input, execution| {
        let evidence = PartialAggregateInputEvidence {
            source_chunk_fingerprint: partial_aggregate_source_chunk_fingerprint(std::slice::from_ref(chunk)),
            source_adjacent_partial: stream.has_stages(),
            source_chunk_row_group_count: observed_chunk_row_group_count(chunk)
                .unwrap_or_else(|| chunk.row_group_count()) as u64,
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
        let stage = super::start_aggregate_chunk_stage(
            execution,
            node_id,
            chunk.chunk_index,
            attempt,
            input.frame.num_rows() as u64,
        );
        match execute_compiled_bounded_partial_aggregate_from_frame_with_retry(
            execution,
            node_id,
            aggregate,
            input_schema,
            output_schema,
            input,
            Some(evidence),
        ) {
            Ok(outputs) => {
                stage.finish(
                    Some(outputs.iter().map(|partial| partial.frame.num_rows() as u64).sum()),
                    Some(outputs.iter().map(CommittedPartial::device_bytes).sum()),
                );
                partials.extend(outputs);
            },
            Err(error) => {
                stage.fail(&error);
                return Err(error);
            },
        }
        Ok(())
    })?;
    Ok(AggregateChunkOutput { partials, source_row_groups })
}

/// Partial-aggregate accumulator for one compiled aggregate stream.
///
/// Partitioned shapes retain their existing incremental owner. Other
/// mergeable aggregates use a stable binary reduction, retaining at most one
/// normalized state per level; grouping sets keep their existing final merge.
/// Which accumulator holds the partials of the stream is fixed at construction (`new`):
///
/// - `Incremental` when the aggregate has no grouping sets and the capability decision selects
///   the partitioned merge for the final stage.
/// - `Folded` for other aggregates that either group by a key or expect more than eight partials.
/// - `Retained` for a few global partials and for grouping sets.
pub(in crate::exec::bounded::aggregate) enum CompiledStreamPartials<'a> {
    Retained {
        node_id: NodeId,
        aggregate: &'a AggregateNode,
        input_schema: &'a ArrowSchemaRef,
        output_schema: &'a ArrowSchemaRef,
        partials: Vec<CommittedPartial>,
    },
    Folded {
        node_id: NodeId,
        aggregate: &'a AggregateNode,
        input_schema: &'a ArrowSchemaRef,
        output_schema: &'a ArrowSchemaRef,
        levels: Vec<Option<CommittedPartial>>,
    },
    Incremental(Box<CompiledIncrementalPartitionedMerge<'a>>),
}

impl<'a> CompiledStreamPartials<'a> {
    /// Chooses the accumulator for a stream expected to yield `expected_partials` partials.
    ///
    /// `expected_input_rows` is the producer's declared row total when it has one. Only the
    /// incremental merge reads either value, and only to size its first partition count.
    pub(in crate::exec::bounded::aggregate) fn new(
        execution: &BoundedExecutionAttempt<'_>,
        node_id: NodeId,
        aggregate: &'a AggregateNode,
        input_schema: &'a ArrowSchemaRef,
        output_schema: &'a ArrowSchemaRef,
        expected_partials: usize,
        expected_input_rows: Option<u64>,
    ) -> Result<Self> {
        if should_execute_incremental_partitioned_final_aggregate_merge(aggregate, input_schema) {
            return CompiledIncrementalPartitionedMerge::new(
                execution.context(),
                node_id,
                aggregate,
                input_schema,
                output_schema,
                expected_partials,
                expected_input_rows,
            )
            .map(|merge| Self::Incremental(Box::new(merge)));
        }
        // A few global states fit in a single bounded final merge. Keeping
        // their original partials also preserves its one-attempt observation.
        if aggregate.spec.grouping_sets.is_none() && (!aggregate.spec.group_exprs.is_empty() || expected_partials > 8) {
            return Ok(Self::Folded { node_id, aggregate, input_schema, output_schema, levels: Vec::new() });
        }
        Ok(Self::Retained {
            node_id,
            aggregate,
            input_schema,
            output_schema,
            partials: Vec::with_capacity(expected_partials),
        })
    }

    /// Takes ownership of one chunk's partials and folds them into the accumulator.
    ///
    /// An error is terminal for the aggregate: earlier partials of the call may already be merged,
    /// so the chunk cannot be replayed. In `Folded`, each partial is first normalized to merge
    /// state so that raw partials and merged states never share a concatenation. Equal levels merge
    /// as a binary carry. At most one state per level is retained, and the merge order follows the
    /// order of `extend` calls, independent of lane completion order.
    pub(in crate::exec::bounded::aggregate) fn extend(
        &mut self,
        execution: &mut BoundedExecutionAttempt<'_>,
        chunk_partials: Vec<CommittedPartial>,
    ) -> Result<()> {
        match self {
            Self::Folded { node_id, aggregate, input_schema, output_schema, levels } => {
                for partial in chunk_partials {
                    let mut state = normalize_compiled_aggregate_partial(
                        execution,
                        *node_id,
                        aggregate,
                        input_schema,
                        output_schema,
                        partial,
                    )?;
                    let mut level = 0;
                    loop {
                        if level == levels.len() {
                            levels.push(Some(state));
                            break;
                        }
                        let Some(earlier) = levels[level].take() else {
                            levels[level] = Some(state);
                            break;
                        };
                        execution.metrics().record_final_merge_attempt(2);
                        state = execute_compiled_final_merge_group_attempt(
                            execution,
                            *node_id,
                            aggregate,
                            input_schema,
                            output_schema,
                            &[earlier, state],
                        )?;
                        execution.metrics().record_final_merge_level();
                        level += 1;
                    }
                }
                Ok(())
            },
            Self::Retained { partials, .. } => {
                partials.extend(chunk_partials);
                Ok(())
            },
            Self::Incremental(merge) => {
                for partial in chunk_partials {
                    merge.push(execution, partial)?;
                }
                Ok(())
            },
        }
    }

    /// Merges what remains and returns the final output.
    ///
    /// `Folded` merges its occupied levels, from the earliest input interval to the latest, in one
    /// final merge. `Retained` merges every partial, or aggregates an empty input when none
    /// arrived. `Incremental` finishes its partitions and either concatenates them or, when
    /// `preserve_output_stream` is set, returns them as a lazy stream finalized one partition per
    /// pull.
    pub(in crate::exec::bounded::aggregate) fn finish(
        self,
        execution: &mut BoundedExecutionAttempt<'_>,
        preserve_output_stream: bool,
    ) -> Result<CompiledGpuResource> {
        match self {
            Self::Folded { node_id, aggregate, input_schema, output_schema, levels } => {
                // Larger occupied levels contain the earliest input interval.
                // The binary carry bounds retained state to one per usize bit
                // and makes merge order independent of lane completion order.
                let partials = levels.into_iter().rev().flatten().collect();
                execute_compiled_bounded_final_aggregate_merge(
                    execution,
                    node_id,
                    aggregate,
                    input_schema,
                    output_schema,
                    partials,
                )
                .map(CompiledGpuResource::new)
            },
            Self::Retained { node_id, aggregate, input_schema, output_schema, partials } => {
                if partials.is_empty() {
                    return execute_compiled_bounded_aggregate_from_empty_input(
                        execution,
                        node_id,
                        aggregate,
                        input_schema,
                        output_schema,
                    )
                    .map(CompiledGpuResource::new);
                }
                execute_compiled_bounded_final_aggregate_merge(
                    execution,
                    node_id,
                    aggregate,
                    input_schema,
                    output_schema,
                    partials,
                )
                .map(CompiledGpuResource::new)
            },
            Self::Incremental(merge) if preserve_output_stream => {
                merge.finish_stream(execution).map(CompiledGpuResource::frame_stream_payload)
            },
            Self::Incremental(merge) => merge.finish(execution).map(CompiledGpuResource::new),
        }
    }
}

/// GPU-native statistical aggregate over resident frames: one statistical partial per frame, merged
/// in one step by `execute_compiled_bounded_statistical_final_merge_from_partials`. All partials
/// are retained until the stream ends. The stream's remaining frames are released if a partial
/// fails.
fn execute_compiled_bounded_statistical_aggregate_from_stream(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &ArrowSchemaRef,
    output_schema: &ArrowSchemaRef,
    inputs: &mut ResidentFrameStreamPayload,
) -> Result<ResidentGpuFrame> {
    let mut partials = Vec::with_capacity(inputs.len());
    while let Some(input) = inputs.next_frame(execution) {
        let input = match input {
            Ok(input) => input,
            Err(error) => {
                return Err(error);
            },
        };
        match execute_compiled_bounded_statistical_partial_from_frame(
            execution,
            node_id,
            aggregate,
            input_schema,
            input,
        ) {
            Ok(partial) => partials.push(partial),
            Err(error) => {
                inputs.release_remaining_frames();
                return Err(error);
            },
        }
    }

    if partials.is_empty() {
        return execute_compiled_bounded_aggregate_from_empty_input(
            execution,
            node_id,
            aggregate,
            input_schema,
            output_schema,
        );
    }

    execute_compiled_bounded_statistical_final_merge_from_partials(
        execution,
        node_id,
        aggregate,
        input_schema,
        output_schema,
        partials,
    )
}

/// GPU-native statistical aggregate over a source stream: one statistical partial per chunk through
/// the chunk consumer, then the same single-step merge as the frame stream path.
fn execute_compiled_bounded_statistical_aggregate_from_source_stream(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &ArrowSchemaRef,
    output_schema: &ArrowSchemaRef,
    stream: BoundedJoinLeafLocalParquetSourceStream,
) -> Result<ResidentGpuFrame> {
    let mut partials = Vec::with_capacity(stream.chunk_count());
    let stream_result = stream_bounded_join_leaf_local_parquet_source_chunks(execution, stream, |input, execution| {
        let partial = execute_compiled_bounded_statistical_partial_from_frame(
            execution,
            node_id,
            aggregate,
            input_schema,
            input,
        )?;
        partials.push(partial);
        Ok(())
    });
    stream_result?;

    if partials.is_empty() {
        return execute_compiled_bounded_aggregate_from_empty_input(
            execution,
            node_id,
            aggregate,
            input_schema,
            output_schema,
        );
    }

    execute_compiled_bounded_statistical_final_merge_from_partials(
        execution,
        node_id,
        aggregate,
        input_schema,
        output_schema,
        partials,
    )
}

/// Unions the statistical partials into one frame and merges them under a budget token for
/// `compiled_statistical_aggregate_final_merge`.
fn execute_compiled_bounded_statistical_final_merge_from_partials(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &ArrowSchemaRef,
    output_schema: &ArrowSchemaRef,
    partials: Vec<CommittedPartial>,
) -> Result<ResidentGpuFrame> {
    let merged = execute_bounded_union(execution, node_id, partials)?;
    run_budgeted(
        execution,
        node_id,
        "compiled_statistical_aggregate_final_merge",
        merged.frame,
        |context, input, token, metrics| {
            execute_gpu_native_statistical_final_merge_with_budget(
                context,
                Some(node_id),
                input.into_frame(),
                &aggregate.spec,
                input_schema,
                output_schema,
                token,
                metrics,
            )
        },
    )
}

/// Computes the statistical partial (moment state) of one frame. Consumes `input`.
fn execute_compiled_bounded_statistical_partial_from_frame(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &ArrowSchemaRef,
    input: ResidentGpuFrame,
) -> Result<CommittedPartial> {
    let input_rows = input.frame.num_rows() as u64;
    let input_frame = input.frame;
    let token = match execution_budget_token_for_boundary(
        execution.context(),
        execution.device_budget_bytes(),
        "compiled_statistical_aggregate_partial_state_work",
        "query_engine.bounded.aggregate.statistical_partial.budget_token",
        "failed to create bounded native statistical partial aggregate budget token",
    ) {
        Ok(token) => token,
        Err(error) => {
            return Err(error);
        },
    };
    let output = execute_gpu_native_statistical_partial_aggregate_with_budget(
        execution.context(),
        Some(node_id),
        input_frame.into_frame(),
        &aggregate.spec,
        input_schema,
        &token,
        execution.metrics(),
    );
    let frame = match output {
        Ok(frame) => frame,
        Err(error) => {
            return Err(error);
        },
    };
    let partial_bytes = match observe_device_memory_bytes(&frame, execution.metrics()) {
        Ok(bytes) => bytes,
        Err(error) => {
            return Err(error);
        },
    };
    execution.metrics().record_partial_aggregate(input_rows, frame.num_rows() as u64, partial_bytes);
    Ok(ResidentGpuFrame::with_device_bytes(frame, partial_bytes))
}

/// DISTINCT aggregates cannot merge per-chunk states. The frames are concatenated and aggregated in
/// one stage. A single frame skips the concatenation.
fn execute_compiled_bounded_distinct_aggregate_from_stream(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &ArrowSchemaRef,
    output_schema: &ArrowSchemaRef,
    inputs: Vec<ResidentGpuFrame>,
) -> Result<ResidentGpuFrame> {
    let mut frames = inputs;
    if frames.len() == 1 {
        let input = frames.pop().expect("single frame checked");
        let input_bytes = input.device_bytes();
        return execute_compiled_bounded_aggregate_from_frame(
            execution,
            node_id,
            aggregate,
            input_schema,
            output_schema,
            input.frame,
            input_bytes,
        );
    }

    let first = frames.remove(0);
    let first_frame = first.frame.into_frame();
    let concatenated = {
        let token = match execution_budget_token_for_boundary(
            execution.context(),
            execution.device_budget_bytes(),
            "compiled_aggregate_distinct_stream_concat",
            "query_engine.bounded.aggregate.distinct_stream_concat.budget_token",
            "failed to create bounded native distinct aggregate concat budget token",
        ) {
            Ok(token) => token,
            Err(error) => {
                return Err(error);
            },
        };
        let other_frame_refs = frames.iter().map(|frame| frame.frame.frame()).collect::<Vec<_>>();
        first_frame.on(token.exec()).concat_many(&other_frame_refs).map_err(|error| {
            attach_budget_token_failure_facts(
                cudf_query_error(
                    "query_engine.bounded.aggregate.distinct_stream_concat",
                    "failed to concatenate distinct aggregate stream inputs",
                    error,
                ),
                &token,
            )
        })
    };
    // The distinct aggregate below allocates, so the concatenated inputs cannot wait for end of scope.
    drop(frames);
    let concatenated = concatenated.map(GpuDataFrame::new)?;
    observe_device_memory_bytes(&concatenated, execution.metrics())?;
    run_budgeted(execution, node_id, "compiled_aggregate_distinct", concatenated, |context, input, token, metrics| {
        execute_aggregate_with_budget(context, aggregate, input_schema, input, output_schema, token, metrics)
    })
}

/// Computes the partial state of one resident frame with no source evidence and no proactive split.
/// Used for the partitions of a hash-partitioned input (`partitioned.rs`).
pub(in crate::exec::bounded::aggregate) fn execute_compiled_bounded_partial_aggregate_from_frame(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &ArrowSchemaRef,
    output_schema: &ArrowSchemaRef,
    input: ResidentGpuFrame,
) -> Result<CommittedPartial> {
    execute_compiled_bounded_partial_aggregate_from_frame_with_evidence(
        execution,
        node_id,
        aggregate,
        input_schema,
        output_schema,
        input,
        None,
    )
}

/// Computes the partial state of one frame, as one or more committed partials.
///
/// A frame whose partial-state bound exceeds the device budget is split in halves first and can
/// therefore yield more than one partial. The final merge treats them as extra chunks.
fn execute_compiled_bounded_partial_aggregate_from_frame_with_retry(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &ArrowSchemaRef,
    output_schema: &ArrowSchemaRef,
    input: ResidentGpuFrame,
    input_evidence: Option<PartialAggregateInputEvidence>,
) -> Result<Vec<CommittedPartial>> {
    execute_compiled_bounded_partial_aggregate_from_frame_with_retry_budget(
        execution,
        node_id,
        aggregate,
        input_schema,
        output_schema,
        input,
        compiled_aggregate_frame_retry_budget(execution.context()),
        input_evidence,
    )
}

/// Splits `input` in halves while its partial-state upper bound exceeds the device budget and
/// `retry_budget` remains, then computes the partial of each piece.
///
/// The sizing bound alone triggers the split, with no allocation failure, retry counter or backoff
/// involved (see `sizing.rs`). Each level of splitting spends one unit of `retry_budget`, and a
/// frame with no bound runs whole. Partials come back in input row order. An error from either half
/// drops the partials already produced for the other.
#[expect(clippy::too_many_arguments, reason = "partial work preserves its source evidence through split retries")]
fn execute_compiled_bounded_partial_aggregate_from_frame_with_retry_budget(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &ArrowSchemaRef,
    output_schema: &ArrowSchemaRef,
    input: ResidentGpuFrame,
    retry_budget: usize,
    input_evidence: Option<PartialAggregateInputEvidence>,
) -> Result<Vec<CommittedPartial>> {
    let upper_bound_bytes = match partial_aggregate_work_reservation_upper_bound_bytes(
        execution,
        aggregate,
        input_schema,
        output_schema,
        &input.frame,
    ) {
        Ok(bytes) => bytes,
        Err(error) => {
            return Err(error);
        },
    };
    if should_split_compiled_aggregate_stream_frame(execution, &input, upper_bound_bytes, retry_budget) {
        // A proactive split has no allocation failure behind it: no retry
        // counter, no backoff, only the cancellation boundary.
        execution.context().check_cancellation("query_engine.exec.bounded.compiled_aggregate_stream_frame_split")?;
        let (left, right) = split_compiled_aggregate_stream_frame(execution, node_id, input)?;
        let mut partials = match execute_compiled_bounded_partial_aggregate_from_frame_with_retry_budget(
            execution,
            node_id,
            aggregate,
            input_schema,
            output_schema,
            left,
            retry_budget.saturating_sub(1),
            input_evidence,
        ) {
            Ok(partials) => partials,
            Err(error) => {
                return Err(error);
            },
        };
        match execute_compiled_bounded_partial_aggregate_from_frame_with_retry_budget(
            execution,
            node_id,
            aggregate,
            input_schema,
            output_schema,
            right,
            retry_budget.saturating_sub(1),
            input_evidence,
        ) {
            Ok(right_partials) => {
                partials.extend(right_partials);
                return Ok(partials);
            },
            Err(error) => {
                return Err(error);
            },
        }
    }
    execute_compiled_bounded_partial_aggregate_from_frame_with_evidence(
        execution,
        node_id,
        aggregate,
        input_schema,
        output_schema,
        input,
        input_evidence,
    )
    .map(|partial| vec![partial])
}

/// One partial aggregate under a budget token, with partial-aggregate metrics (input rows, output
/// rows, state bytes). `input_evidence` is present for source chunks and selects the
/// `chunk_operator` token boundary; otherwise the boundary is
/// `compiled_aggregate_partial_state_work`.
fn execute_compiled_bounded_partial_aggregate_from_frame_with_evidence(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &ArrowSchemaRef,
    output_schema: &ArrowSchemaRef,
    input: ResidentGpuFrame,
    input_evidence: Option<PartialAggregateInputEvidence>,
) -> Result<CommittedPartial> {
    let input_rows = input.frame.num_rows() as u64;
    let token = match execution_budget_token_for_boundary(
        execution.context(),
        execution.device_budget_bytes(),
        if input_evidence.is_some() { "chunk_operator" } else { "compiled_aggregate_partial_state_work" },
        "query_engine.bounded.aggregate.compiled_stream_partial.budget_token",
        "failed to create bounded native compiled stream aggregate budget token",
    ) {
        Ok(token) => token,
        Err(error) => {
            return Err(error);
        },
    };
    let output = execute_partial_aggregate_with_budget(
        execution.context(),
        Some(node_id),
        aggregate,
        input_schema,
        output_schema,
        input.frame,
        &token,
        input_evidence,
        execution.metrics(),
    );
    let frame = match output {
        Ok(frame) => frame,
        Err(error) => {
            return Err(error);
        },
    };
    let partial_bytes = match observe_device_memory_bytes(&frame, execution.metrics()) {
        Ok(bytes) => bytes,
        Err(error) => {
            return Err(error);
        },
    };
    execution.metrics().record_partial_aggregate(input_rows, frame.num_rows() as u64, partial_bytes);
    Ok(ResidentGpuFrame::with_device_bytes(frame, partial_bytes))
}

/// Splits a frame proactively when its partial state can outgrow the budget.
/// Without a bound the frame runs whole, and a frame of one row never splits.
fn should_split_compiled_aggregate_stream_frame(
    execution: &mut BoundedExecutionAttempt<'_>,
    input: &ResidentGpuFrame,
    upper_bound_bytes: Option<u64>,
    retry_budget: usize,
) -> bool {
    retry_budget > 0
        && input.frame.num_rows() > 1
        && upper_bound_bytes.is_some_and(|bytes| bytes > execution.device_budget_bytes())
}

/// Levels of proactive splitting allowed: one less than the context's retry attempt limit.
fn compiled_aggregate_frame_retry_budget(context: &ExecutionContext) -> usize {
    RetryLimits::of(context).max_attempts - 1
}

/// Slices a frame at its row midpoint into two owned frames and observes their device bytes. The
/// source frame is released when the call returns. A frame of fewer than two rows is an invariant
/// violation.
fn split_compiled_aggregate_stream_frame(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    input: ResidentGpuFrame,
) -> Result<(ResidentGpuFrame, ResidentGpuFrame)> {
    let rows = input.frame.num_rows();
    let split_at = rows / 2;
    if split_at == 0 || split_at == rows {
        return Err(bounded_invariant_violation(
            "compiled aggregate stream frame cannot be split further",
            "compiled_aggregate_stream_frame_retry_unsplittable",
        )
        .with_fact("node_id", node_id)
        .with_fact("input_rows", rows));
    }
    let source_frame = input.frame.into_frame();
    let exec_runtime = execution.context().runtime()?;
    let overrides = exec_runtime.exec();
    let left = match source_frame.on(overrides).slice(0, split_at) {
        Ok(left) => left,
        Err(error) => {
            return Err(cudf_query_error(
                "query_engine.bounded.aggregate.stream_frame_split",
                "failed to split compiled aggregate stream frame",
                error,
            )
            .with_fact("node_id", node_id)
            .with_fact("split_side", "left")
            .with_fact("input_rows", rows)
            .with_fact("split_at", split_at));
        },
    };
    let right_len = rows.saturating_sub(split_at);
    let right = match source_frame.on(overrides).slice(split_at, right_len) {
        Ok(right) => right,
        Err(error) => {
            return Err(cudf_query_error(
                "query_engine.bounded.aggregate.stream_frame_split",
                "failed to split compiled aggregate stream frame",
                error,
            )
            .with_fact("node_id", node_id)
            .with_fact("split_side", "right")
            .with_fact("input_rows", rows)
            .with_fact("split_at", split_at));
        },
    };
    let left = GpuDataFrame::new(left);
    let right = GpuDataFrame::new(right);
    let left_bytes = observe_device_memory_bytes(&left, execution.metrics())?;
    let right_bytes = observe_device_memory_bytes(&right, execution.metrics())?;
    Ok((ResidentGpuFrame::with_device_bytes(left, left_bytes), ResidentGpuFrame::with_device_bytes(right, right_bytes)))
}
