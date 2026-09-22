use std::time::Instant;

use super::{final_merge::partition_final_merge_partial, partitioned::partitioned_aggregate_partition_count, *};
use crate::{
    exec::{
        bounded::{
            chunks::{
                bounded_source_adjacent_chunk_batches_for_work_width, bounded_source_adjacent_chunk_batching_enabled,
                execute_source_adjacent_batch_uncommitted,
            },
            compiled::{CompiledGpuResource, lanes::AggregateLaneDispatch},
        },
        retry::backoff_before_retry,
    },
    naming::unique_alias_with_suffix,
    observability::{
        RuntimeOperatorStageGuard, RuntimeOperatorStageIdentity, RuntimeOperatorStageKind, stage_context_from_execution,
    },
    rapids_backend::rapids_analysis_context,
    util::sum_optional_u64,
};

/// Source and aggregate ownership needed by a batch worker; downstream plan
/// stages remain with the compiled coordinator.
struct OwnedSourceAdjacentAggregate {
    source_id: NodeId,
    source: LocalParquetSource,
    pre_filter_projections: Vec<(ProjectionNode, ArrowSchemaRef)>,
    filter: Option<FilterNode>,
    post_filter_projections: Vec<(ProjectionNode, ArrowSchemaRef)>,
    aggregate_id: NodeId,
    input_schema: ArrowSchemaRef,
    aggregate: AggregateNode,
    output_schema: ArrowSchemaRef,
}

impl OwnedSourceAdjacentAggregate {
    fn new(bounded: &BoundedPlan<'_>) -> Self {
        Self {
            source_id: bounded.source_id,
            source: bounded.source.clone(),
            pre_filter_projections: bounded
                .pre_filter_projections
                .iter()
                .map(|(projection, schema)| ((*projection).clone(), Arc::clone(schema)))
                .collect(),
            filter: bounded.filter.cloned(),
            post_filter_projections: bounded
                .post_filter_projections
                .iter()
                .map(|(projection, schema)| ((*projection).clone(), Arc::clone(schema)))
                .collect(),
            aggregate_id: bounded.aggregate_id,
            input_schema: Arc::clone(bounded.aggregate_input_schema),
            aggregate: bounded.aggregate.0.clone(),
            output_schema: Arc::clone(bounded.aggregate.1),
        }
    }

    fn plan(&self) -> BoundedPlan<'_> {
        BoundedPlan {
            source_id: self.source_id,
            source: &self.source,
            pre_filter_projections: self.pre_filter_projections.iter().map(|(node, schema)| (node, schema)).collect(),
            filter: self.filter.as_ref(),
            post_filter_projections: self.post_filter_projections.iter().map(|(node, schema)| (node, schema)).collect(),
            aggregate_id: self.aggregate_id,
            aggregate_input_schema: &self.input_schema,
            aggregate: (&self.aggregate, &self.output_schema),
            post_aggregate_stages: Vec::new(),
            sort: None,
            post_sort_projections: Vec::new(),
            limit: None,
        }
    }
}

pub(in crate::exec::bounded) fn execute_source_aggregate_with_dispatch(
    execution: &mut BoundedExecutionAttempt<'_>,
    bounded: &BoundedPlan<'_>,
    chunks: &[SourceChunk],
    dispatch: Option<&AggregateLaneDispatch>,
    partitioned: bool,
) -> Result<CompiledGpuResource> {
    if partitioned {
        return execute_partitioned_source_adjacent_aggregate(execution, bounded, chunks, dispatch);
    }
    let batches = if bounded_source_adjacent_chunk_batching_enabled(execution.context(), bounded, chunks) {
        bounded_source_adjacent_chunk_batches_for_work_width(
            execution.context(),
            bounded,
            chunks,
            dispatch.map_or(1, AggregateLaneDispatch::width),
        )
    } else {
        singleton_source_chunk_batches(chunks)
    };
    let partials = super::stream::CompiledStreamPartials::new(
        execution,
        bounded.aggregate_id,
        bounded.aggregate.0,
        bounded.aggregate_input_schema,
        bounded.aggregate.1,
        batches.len(),
        sum_optional_u64(chunks.iter().map(|chunk| chunk.known_row_count)),
    )?;
    let mut state = (batches.into_iter(), partials);
    if let Some(dispatch) = dispatch {
        let owned = Arc::new(OwnedSourceAdjacentAggregate::new(bounded));
        dispatch.execute_work(
            execution,
            &mut state,
            |state, _| Ok(state.0.next()),
            move |execution, batch| execute_source_adjacent_batch_uncommitted(execution, &owned.plan(), batch),
            |state, execution, output| {
                state.1.extend(execution, output.partials)?;
                execution.metrics().record_source_chunk_commit_count(output.chunk_count, output.row_groups);
                Ok(())
            },
        )?;
    } else {
        for batch in state.0 {
            let output = execute_source_adjacent_batch_uncommitted(execution, bounded, batch)?;
            execution.context().check_cancellation("query_engine.exec.bounded.source_adjacent_batch_commit")?;
            state.1.extend(execution, output.partials)?;
            execution.metrics().record_source_chunk_commit_count(output.chunk_count, output.row_groups);
        }
    }
    state.1.finish(execution, false)
}

struct SourceAdjacentPartitionInput {
    chunks: Vec<SourceChunk>,
    partial: CommittedPartial,
    source_row_group_count: u64,
}

struct PartitionedSourceBatchWork {
    batch: SourceChunkBatch,
    pending: Option<Vec<Vec<SourceAdjacentPartitionInput>>>,
    partition_count: Option<usize>,
    local_reduction: bool,
    estimated_input_bytes: u64,
    min_partition_count: usize,
}

struct PartitionedSourceBatchOutput {
    pending: Option<Vec<Vec<SourceAdjacentPartitionInput>>>,
    partition_count: Option<usize>,
    outputs: Vec<SourceAdjacentAlignedPartial>,
    samples: Vec<SourceAdjacentRawBatchSample>,
    commits: Vec<(u64, u64)>,
}

/// Partial aggregate output that remains hash-aligned to a group-key partition.
struct SourceAdjacentAlignedPartial {
    partition_index: usize,
    partition_count: usize,
    key_columns: Vec<String>,
    partial: CommittedPartial,
}

enum SourceAdjacentPartitionedFinal<'a> {
    Incremental { merge: Box<CompiledIncrementalPartitionedMerge<'a>>, aligned_key_columns: Option<Vec<String>> },
    Stream(super::stream::CompiledStreamPartials<'a>),
}

impl<'a> SourceAdjacentPartitionedFinal<'a> {
    fn new(
        execution: &BoundedExecutionAttempt<'_>,
        bounded: &'a BoundedPlan<'_>,
        partitioner: &SourceAdjacentAggregatePartitioner,
        chunk_count: usize,
        expected_input_rows: Option<u64>,
    ) -> Result<Self> {
        if let Some(key_columns) = partitioner.group_key_columns() {
            return CompiledIncrementalPartitionedMerge::new_aligned(
                execution.context(),
                bounded.aggregate_id,
                bounded.aggregate.0,
                bounded.aggregate_input_schema,
                bounded.aggregate.1,
                key_columns,
            )
            .map(|merge| Self::Incremental {
                merge: Box::new(merge),
                aligned_key_columns: Some(key_columns.to_vec()),
            });
        }
        if should_execute_incremental_partitioned_final_aggregate_merge(
            execution.context(),
            bounded.aggregate.0,
            bounded.aggregate_input_schema,
        ) {
            return CompiledIncrementalPartitionedMerge::new(
                execution.context(),
                bounded.aggregate_id,
                bounded.aggregate.0,
                bounded.aggregate_input_schema,
                bounded.aggregate.1,
                chunk_count,
                expected_input_rows,
            )
            .map(|merge| Self::Incremental { merge: Box::new(merge), aligned_key_columns: None });
        }
        super::stream::CompiledStreamPartials::new(
            execution,
            bounded.aggregate_id,
            bounded.aggregate.0,
            bounded.aggregate_input_schema,
            bounded.aggregate.1,
            chunk_count,
            expected_input_rows,
        )
        .map(Self::Stream)
    }

    fn accept(
        &mut self,
        execution: &mut BoundedExecutionAttempt<'_>,
        outputs: Vec<SourceAdjacentAlignedPartial>,
    ) -> Result<()> {
        match self {
            Self::Incremental { merge, aligned_key_columns: Some(_) } => {
                for output in outputs {
                    merge.push_aligned(
                        execution,
                        output.partition_index,
                        output.partition_count,
                        &output.key_columns,
                        output.partial,
                    )?;
                }
                Ok(())
            },
            Self::Incremental { merge, aligned_key_columns: None } => {
                for output in outputs {
                    merge.push(execution, output.partial)?;
                }
                Ok(())
            },
            Self::Stream(partials) => {
                partials.extend(execution, outputs.into_iter().map(|output| output.partial).collect())
            },
        }
    }

    fn finish(
        self,
        execution: &mut BoundedExecutionAttempt<'_>,
        bounded: &BoundedPlan<'_>,
    ) -> Result<CompiledGpuResource> {
        match self {
            Self::Incremental { merge, aligned_key_columns } => {
                execute_incremental_source_adjacent_final_aggregate_merge(
                    execution,
                    bounded,
                    *merge,
                    aligned_key_columns.as_deref(),
                )
                .map(CompiledGpuResource::new)
            },
            Self::Stream(partials) => partials.finish(execution, false),
        }
    }
}

#[derive(Clone, Debug)]
enum SourceAdjacentAggregatePartitioner {
    RowBalanced,
    GroupKey { key_columns: Vec<String> },
}

impl SourceAdjacentAggregatePartitioner {
    fn label(&self) -> &'static str {
        match self {
            Self::RowBalanced => "row_balanced",
            Self::GroupKey { .. } => "group_key",
        }
    }

    fn key_count(&self) -> Option<u64> {
        match self {
            Self::RowBalanced => None,
            Self::GroupKey { key_columns } => Some(key_columns.len() as u64),
        }
    }

    fn group_key_columns(&self) -> Option<&[String]> {
        match self {
            Self::RowBalanced => None,
            Self::GroupKey { key_columns } => Some(key_columns.as_slice()),
        }
    }

    fn retain_empty_partition(&self, partition: usize) -> bool {
        match self {
            Self::RowBalanced => true,
            Self::GroupKey { .. } => partition == 0,
        }
    }
}

/// Decoded width of the read columns before any observation: the sum of the
/// fixed-width Arrow byte widths and the count of variable-width columns.
fn cold_read_column_widths(source: &LocalParquetSource, read_columns: &[String]) -> (u64, u64) {
    let schema = source.output_schema();
    read_columns.iter().fold((0u64, 0u64), |(fixed, variable), column| {
        match schema.field_with_name(column).ok().and_then(|field| field.data_type().primitive_width()) {
            Some(width) => (fixed.saturating_add(width as u64), variable),
            None => (fixed, variable + 1),
        }
    })
}

/// One completed raw source batch, as observed rather than estimated.
///
/// Decoded bytes come from the reader's output frame and staged bytes from the
/// frame after the source-adjacent filter and projections, so the next batch
/// is sized from what this source actually decodes to instead of its encoded
/// size. Partial rows are known only when the batch produced partial output;
/// partition occupancy is known only when the batch was hash-partitioned.
#[derive(Debug, Clone, Copy)]
struct SourceAdjacentRawBatchSample {
    rows: u64,
    decoded_bytes: u64,
    staged_bytes: u64,
    partial_rows: Option<u64>,
    partial_bytes: u64,
    non_empty_partitions: Option<usize>,
    partition_count: Option<usize>,
}

/// Working-set feedback across the raw batches of one source-adjacent aggregate.
///
/// The feedback only chooses the row cap of the next batch and whether it is
/// reduced locally; the query-domain allocator remains the sole cap enforcer.
#[derive(Default)]
struct SourceAdjacentRawBatchFeedback {
    rows: u64,
    decoded_bytes: u64,
    staged_bytes: u64,
    last: Option<SourceAdjacentRawBatchSample>,
}

impl SourceAdjacentRawBatchFeedback {
    fn observe(&mut self, sample: SourceAdjacentRawBatchSample) {
        self.rows = self.rows.saturating_add(sample.rows);
        self.decoded_bytes = self.decoded_bytes.saturating_add(sample.decoded_bytes);
        self.staged_bytes = self.staged_bytes.saturating_add(sample.staged_bytes);
        self.last = Some(sample);
    }

    /// Row cap for the next raw batch, or `None` when neither an observation
    /// nor footer row counts can bound the decoded working set.
    fn next_batch_row_cap(
        &self,
        policy: BoundedSourceBatchingPolicy,
        device_budget_bytes: u64,
        source: &LocalParquetSource,
        remaining_chunks: &[SourceChunk],
    ) -> Option<u64> {
        let target_bytes = policy.source_adjacent_raw_batch_working_set_target_bytes(device_budget_bytes);
        let bytes_per_row = if self.rows > 0 {
            policy.source_adjacent_raw_batch_observed_working_set_bytes_per_row(
                self.decoded_bytes.div_ceil(self.rows),
                self.staged_bytes.div_ceil(self.rows),
            )
        } else {
            let (rows, uncompressed_bytes) = remaining_chunks
                .iter()
                .filter_map(|chunk| Some((chunk.known_row_count?, chunk.known_projected_uncompressed_bytes?)))
                .fold((0u64, 0u64), |(rows, bytes), (chunk_rows, chunk_bytes)| {
                    (rows.saturating_add(chunk_rows), bytes.saturating_add(chunk_bytes))
                });
            if rows == 0 {
                return None;
            }
            let (fixed_width_bytes, variable_width_columns) =
                cold_read_column_widths(source, &remaining_chunks[0].read_columns);
            policy.source_adjacent_raw_batch_cold_working_set_bytes_per_row(
                fixed_width_bytes,
                variable_width_columns,
                uncompressed_bytes.div_ceil(rows),
            )
        };
        Some((target_bytes / bytes_per_row).max(1))
    }

    /// Whether the next batch should be reduced whole before partitioning.
    ///
    /// A locally reduced batch proves its own reduction through its partial
    /// output. A hash-partitioned batch that did not flush proves low key
    /// cardinality through partition occupancy instead: keys hash uniformly,
    /// so a batch that touched only a small fraction of the partitions held
    /// few distinct keys.
    fn local_reduction_effective(&self, policy: BoundedSourceBatchingPolicy, device_budget_bytes: u64) -> bool {
        let Some(sample) = self.last else {
            return false;
        };
        if let Some(partial_rows) = sample.partial_rows {
            return policy.source_adjacent_local_reduction_effective(
                sample.rows,
                partial_rows,
                sample.partial_bytes,
                device_budget_bytes,
            );
        }
        match (sample.non_empty_partitions, sample.partition_count) {
            (Some(non_empty), Some(partition_count)) => {
                policy.source_adjacent_local_reduction_from_partition_occupancy(non_empty, partition_count)
            },
            _ => false,
        }
    }
}

fn execute_partitioned_source_adjacent_aggregate(
    execution: &mut BoundedExecutionAttempt<'_>,
    bounded: &BoundedPlan<'_>,
    chunks: &[SourceChunk],
    dispatch: Option<&AggregateLaneDispatch>,
) -> Result<CompiledGpuResource> {
    let owned = dispatch.map(|_| Arc::new(OwnedSourceAdjacentAggregate::new(bounded)));
    let partitioner = source_adjacent_aggregate_partitioner(bounded);
    let estimated_input_bytes = chunks
        .iter()
        .filter_map(|chunk| chunk.estimated_chunk_bytes.or(chunk.estimated_source_read_bytes))
        .sum::<u64>()
        .max(1);
    let batching_policy = BoundedSourceBatchingPolicy::default();
    let device_budget_bytes = execution.device_budget_bytes();
    let mut partition_count = None;
    let mut partition_inputs: Option<Vec<Vec<SourceAdjacentPartitionInput>>> = None;
    let mut final_merge = SourceAdjacentPartitionedFinal::new(
        execution,
        bounded,
        &partitioner,
        chunks.len(),
        sum_optional_u64(chunks.iter().map(|chunk| chunk.known_row_count)),
    )?;
    let mut feedback = SourceAdjacentRawBatchFeedback::default();

    let mut committed_source_chunks = 0u64;
    let mut committed_row_groups = 0u64;
    let mut remaining_chunks = chunks;
    let mut batch_ordinal = 0usize;
    while !remaining_chunks.is_empty() {
        let row_cap =
            feedback.next_batch_row_cap(batching_policy, device_budget_bytes, bounded.source, remaining_chunks);
        let batch = source_adjacent_next_raw_source_batch(
            execution.context().memory_control_policy(),
            bounded,
            remaining_chunks,
            row_cap,
        );
        remaining_chunks = &remaining_chunks[batch.chunk_count()..];
        let local_reduction = feedback.local_reduction_effective(batching_policy, device_budget_bytes);
        let batch_task_id =
            source_adjacent_aggregate_task_id(bounded.aggregate_id, "source_adjacent_raw_batch", batch_ordinal);
        batch_ordinal += 1;
        let budget_window = execution.metrics().open_budget_window();
        let lane_batches =
            split_partitioned_raw_batch_for_lanes(batch, dispatch.map_or(1, AggregateLaneDispatch::width));
        if lane_batches.len() > 1 {
            let dispatch = dispatch.expect("multiple raw batches require aggregate lanes");
            let owned = Arc::clone(owned.as_ref().expect("lane dispatch owns its source descriptor"));
            let partitioner = partitioner.clone();
            let min_partition_count = source_adjacent_partitioned_aggregate_min_partition_count(chunks.len());
            let mut batches = lane_batches.into_iter();
            let mut samples = Vec::new();
            let result = dispatch.execute_work(
                execution,
                &mut batches,
                |batches, _| Ok(batches.next()),
                move |execution, batch| {
                    process_partitioned_source_adjacent_independent_batch(
                        execution,
                        &owned.plan(),
                        PartitionedSourceBatchWork {
                            batch,
                            pending: None,
                            partition_count,
                            local_reduction,
                            estimated_input_bytes,
                            min_partition_count,
                        },
                        &partitioner,
                    )
                },
                |_, execution, output| {
                    if let Some(count) = output.partition_count {
                        debug_assert!(partition_count.is_none_or(|previous| previous == count));
                        partition_count = Some(count);
                    }
                    final_merge.accept(execution, output.outputs)?;
                    for (chunk_count, row_group_count) in output.commits {
                        committed_source_chunks = committed_source_chunks.saturating_add(chunk_count);
                        committed_row_groups = committed_row_groups.saturating_add(row_group_count);
                    }
                    samples.extend(output.samples);
                    Ok(())
                },
            );
            result?;
            for sample in samples {
                feedback.observe(sample);
            }
            execution.metrics().close_budget_window(&batch_task_id, budget_window);
            continue;
        }
        let work = PartitionedSourceBatchWork {
            batch: lane_batches.into_iter().next().expect("one unsplit raw batch remains"),
            pending: partition_inputs.take(),
            partition_count,
            local_reduction,
            estimated_input_bytes,
            min_partition_count: source_adjacent_partitioned_aggregate_min_partition_count(chunks.len()),
        };
        let result = execute_partitioned_source_adjacent_batch_work(
            execution,
            bounded,
            work,
            &partitioner,
            dispatch,
            owned.as_ref(),
        );
        let output = match result {
            Ok(output) => output,
            Err(error) => {
                return Err(error);
            },
        };
        partition_count = output.partition_count;
        partition_inputs = output.pending;
        final_merge.accept(execution, output.outputs)?;
        for (chunk_count, row_group_count) in output.commits {
            committed_source_chunks = committed_source_chunks.saturating_add(chunk_count);
            committed_row_groups = committed_row_groups.saturating_add(row_group_count);
        }
        for sample in output.samples {
            feedback.observe(sample);
        }
        execution.metrics().close_budget_window(&batch_task_id, budget_window);
    }

    let partition_count = partition_count.unwrap_or(1);
    let final_outputs = execute_partitioned_source_adjacent_final_work(
        execution,
        bounded,
        partition_inputs.take().unwrap_or_default(),
        partition_count,
        &partitioner,
        dispatch,
        owned.as_ref(),
    );
    match final_outputs {
        Ok(outputs) => final_merge.accept(execution, outputs)?,
        Err(error) => {
            return Err(error);
        },
    }
    execution.metrics().record_source_chunk_commit_count(committed_source_chunks, committed_row_groups);
    final_merge.finish(execution, bounded)
}

fn split_partitioned_raw_batch_for_lanes(batch: SourceChunkBatch, width: usize) -> Vec<SourceChunkBatch> {
    if width < 2 || batch.chunk_count() < width {
        return vec![batch];
    }
    let chunks_per_batch = batch.chunk_count().div_ceil(width);
    let mut remaining = batch.chunks.into_iter();
    (0..width)
        .filter_map(|index| {
            let chunks = remaining.by_ref().take(chunks_per_batch).collect::<Vec<_>>();
            if chunks.is_empty() {
                return None;
            }
            let estimated_source_read_bytes =
                sum_optional_u64(chunks.iter().map(|chunk| chunk.estimated_source_read_bytes));
            Some(SourceChunkBatch {
                batch_index: batch.batch_index + index,
                chunks,
                estimated_source_read_bytes,
                source_batching_reason: "aggregate_lane_split".to_string(),
            })
        })
        .collect()
}

fn process_partitioned_source_adjacent_independent_batch(
    execution: &mut BoundedExecutionAttempt<'_>,
    bounded: &BoundedPlan<'_>,
    work: PartitionedSourceBatchWork,
    partitioner: &SourceAdjacentAggregatePartitioner,
) -> Result<PartitionedSourceBatchOutput> {
    let mut output = process_partitioned_source_adjacent_batch(execution, bounded, work, partitioner)?;
    let pending = output.pending.take().unwrap_or_default();
    let final_outputs = process_partitioned_source_adjacent_final_batches(
        execution,
        bounded,
        pending,
        output.partition_count.unwrap_or(1),
        partitioner,
    );
    match final_outputs {
        Ok(final_outputs) => output.outputs.extend(final_outputs),
        Err(error) => {
            release_aligned_partials_reverse(output.outputs);
            return Err(error);
        },
    }
    Ok(output)
}

fn execute_partitioned_source_adjacent_batch_work(
    execution: &mut BoundedExecutionAttempt<'_>,
    bounded: &BoundedPlan<'_>,
    work: PartitionedSourceBatchWork,
    partitioner: &SourceAdjacentAggregatePartitioner,
    dispatch: Option<&AggregateLaneDispatch>,
    owned: Option<&Arc<OwnedSourceAdjacentAggregate>>,
) -> Result<PartitionedSourceBatchOutput> {
    let Some(dispatch) = dispatch else {
        return process_partitioned_source_adjacent_batch(execution, bounded, work, partitioner);
    };
    let owned = Arc::clone(owned.expect("lane dispatch owns its source descriptor"));
    let partitioner = partitioner.clone();
    let mut state = (Some(work), None);
    dispatch.execute_work(
        execution,
        &mut state,
        |state, _| Ok(state.0.take()),
        move |execution, work| process_partitioned_source_adjacent_batch(execution, &owned.plan(), work, &partitioner),
        |state, _, output| {
            state.1 = Some(output);
            Ok(())
        },
    )?;
    Ok(state.1.expect("one partitioned batch completes before feedback is updated"))
}

fn process_partitioned_source_adjacent_batch(
    execution: &mut BoundedExecutionAttempt<'_>,
    bounded: &BoundedPlan<'_>,
    work: PartitionedSourceBatchWork,
    partitioner: &SourceAdjacentAggregatePartitioner,
) -> Result<PartitionedSourceBatchOutput> {
    let PartitionedSourceBatchWork {
        batch,
        mut pending,
        mut partition_count,
        local_reduction,
        estimated_input_bytes,
        min_partition_count,
    } = work;
    let mut outputs = Vec::new();
    let mut samples = Vec::new();
    let mut commits = Vec::new();
    let result = (|| -> Result<()> {
        let raw_inputs = execute_source_adjacent_raw_source_batch(execution, bounded, batch)?;
        for (batch_chunks, raw) in raw_inputs {
            let chunk_index = batch_chunks.first().expect("raw source batch has at least one chunk").chunk_index;
            let current_partition_count = *partition_count.get_or_insert_with(|| {
                partitioned_aggregate_partition_count(execution.context(), estimated_input_bytes)
                    .max(min_partition_count)
            });
            let mut sample = SourceAdjacentRawBatchSample {
                rows: raw.source_rows,
                decoded_bytes: raw.source_bytes,
                staged_bytes: raw.partial.device_bytes(),
                partial_rows: None,
                partial_bytes: 0,
                non_empty_partitions: None,
                partition_count: None,
            };
            let source_row_group_count = raw.source_row_group_count;
            if local_reduction {
                let reduced = execute_source_adjacent_batch_local_reduction(
                    execution,
                    bounded,
                    &batch_chunks,
                    raw,
                    current_partition_count,
                    partitioner,
                )?;
                observe_aligned_partial_outputs(&mut sample, &reduced);
                outputs.extend(reduced);
            } else {
                let partitions = partition_source_adjacent_raw_chunk(
                    execution,
                    bounded.aggregate_id,
                    chunk_index,
                    raw.partial,
                    current_partition_count,
                    partitioner,
                )?;
                sample.non_empty_partitions = Some(
                    partitions
                        .iter()
                        .filter(|partition| partition.as_ref().is_some_and(|partition| partition.frame.num_rows() > 0))
                        .count(),
                );
                sample.partition_count = Some(current_partition_count);
                let inputs = pending.get_or_insert_with(|| (0..current_partition_count).map(|_| Vec::new()).collect());
                for (partition_index, partition) in partitions.into_iter().enumerate() {
                    if let Some(partition) = partition {
                        // Oversized partitions still enter the budgeted partial retry.
                        inputs[partition_index].push(SourceAdjacentPartitionInput {
                            chunks: batch_chunks.clone(),
                            partial: partition,
                            source_row_group_count,
                        });
                    }
                }
                for (partition_index, partition_inputs) in inputs.iter_mut().enumerate() {
                    if !source_adjacent_partition_batch_ready(execution.context(), partition_inputs) {
                        continue;
                    }
                    let reduced = execute_source_adjacent_partition_aggregate_batch(
                        execution,
                        bounded,
                        partition_index,
                        current_partition_count,
                        partitioner,
                        partition_inputs,
                    )?;
                    observe_aligned_partial_outputs(&mut sample, &reduced);
                    outputs.extend(reduced);
                }
            }
            commits.push((batch_chunks.len() as u64, source_row_group_count));
            samples.push(sample);
        }
        Ok(())
    })();
    if let Err(error) = result {
        release_nested_partition_inputs_reverse(pending.take().unwrap_or_default());
        release_aligned_partials_reverse(outputs);
        return Err(error);
    }
    Ok(PartitionedSourceBatchOutput { pending, partition_count, outputs, samples, commits })
}

fn execute_partitioned_source_adjacent_final_work(
    execution: &mut BoundedExecutionAttempt<'_>,
    bounded: &BoundedPlan<'_>,
    pending: Vec<Vec<SourceAdjacentPartitionInput>>,
    partition_count: usize,
    partitioner: &SourceAdjacentAggregatePartitioner,
    dispatch: Option<&AggregateLaneDispatch>,
    owned: Option<&Arc<OwnedSourceAdjacentAggregate>>,
) -> Result<Vec<SourceAdjacentAlignedPartial>> {
    let Some(dispatch) = dispatch else {
        return process_partitioned_source_adjacent_final_batches(
            execution,
            bounded,
            pending,
            partition_count,
            partitioner,
        );
    };
    let owned = Arc::clone(owned.expect("lane dispatch owns its source descriptor"));
    let partitioner = partitioner.clone();
    let mut state = (Some(pending), None);
    dispatch.execute_work(
        execution,
        &mut state,
        |state, _| Ok(state.0.take()),
        move |execution, pending| {
            process_partitioned_source_adjacent_final_batches(
                execution,
                &owned.plan(),
                pending,
                partition_count,
                &partitioner,
            )
        },
        |state, _, output| {
            state.1 = Some(output);
            Ok(())
        },
    )?;
    Ok(state.1.expect("final partition batches complete before final merge"))
}

fn process_partitioned_source_adjacent_final_batches(
    execution: &mut BoundedExecutionAttempt<'_>,
    bounded: &BoundedPlan<'_>,
    pending: Vec<Vec<SourceAdjacentPartitionInput>>,
    partition_count: usize,
    partitioner: &SourceAdjacentAggregatePartitioner,
) -> Result<Vec<SourceAdjacentAlignedPartial>> {
    let mut outputs = Vec::new();
    for (partition_index, mut inputs) in pending.into_iter().enumerate() {
        if inputs.is_empty() {
            continue;
        }
        match execute_source_adjacent_partition_aggregate_batch(
            execution,
            bounded,
            partition_index,
            partition_count,
            partitioner,
            &mut inputs,
        ) {
            Ok(mut partials) => outputs.append(&mut partials),
            Err(error) => {
                release_partition_inputs_reverse(inputs);
                release_aligned_partials_reverse(outputs);
                return Err(error);
            },
        }
    }
    Ok(outputs)
}

fn observe_aligned_partial_outputs(
    sample: &mut SourceAdjacentRawBatchSample,
    outputs: &[SourceAdjacentAlignedPartial],
) {
    if outputs.is_empty() {
        return;
    }
    let rows = outputs.iter().map(|output| output.partial.frame.num_rows() as u64).sum::<u64>();
    let bytes = outputs.iter().map(|output| output.partial.device_bytes()).sum::<u64>();
    sample.partial_rows = Some(sample.partial_rows.unwrap_or(0).saturating_add(rows));
    sample.partial_bytes = sample.partial_bytes.saturating_add(bytes);
}

const SOURCE_ADJACENT_LOCAL_REDUCTION_STAGE: &str = "source_adjacent_local_reduction";

/// Partially aggregates one staged batch whole, then hash-partitions the
/// (already small) partial so it stays aligned with the group-key partitions.
///
/// A strongly reducing batch keeps only a few group rows, so slicing the whole
/// staged batch into partitions first would only add two copies of it.
fn execute_source_adjacent_batch_local_reduction(
    execution: &mut BoundedExecutionAttempt<'_>,
    bounded: &BoundedPlan<'_>,
    chunks: &[SourceChunk],
    raw: RawSourceInput,
    partition_count: usize,
    partitioner: &SourceAdjacentAggregatePartitioner,
) -> Result<Vec<SourceAdjacentAlignedPartial>> {
    let input_evidence = PartialAggregateInputEvidence {
        source_adjacent_partition_strategy: "local_reduction",
        source_adjacent_max_partition_fan_in: Some(1),
        ..source_adjacent_partitioned_aggregate_input_evidence(
            chunks,
            raw.source_row_group_count,
            partitioner,
            partition_count,
            1,
            1,
        )
    };
    let partials = execute_source_adjacent_partition_aggregate_with_retry(
        execution,
        bounded,
        SOURCE_ADJACENT_LOCAL_REDUCTION_STAGE,
        0,
        raw.partial,
        input_evidence,
        source_adjacent_partition_aggregate_retry_budget(execution.context()),
    )?;
    let Some(key_columns) = partitioner.group_key_columns() else {
        return Ok(partials
            .into_iter()
            .map(|partial| SourceAdjacentAlignedPartial {
                partition_index: 0,
                partition_count,
                key_columns: Vec::new(),
                partial,
            })
            .collect());
    };
    let output_key_columns = lower_final_aggregate_spec(&bounded.aggregate.0.spec)?.key_columns;
    let mut outputs = Vec::new();
    for (partial_index, partial) in partials.into_iter().enumerate() {
        let partitions = match partition_final_merge_partial(
            execution,
            bounded.aggregate_id,
            partial_index,
            partial,
            &output_key_columns,
            partition_count,
        ) {
            Ok(partitions) => partitions,
            Err(error) => {
                release_aligned_partials_reverse(outputs);
                return Err(error);
            },
        };
        for (partition_index, partition) in partitions.into_iter().enumerate() {
            if partition.frame.num_rows() == 0 && !partitioner.retain_empty_partition(partition_index) {
                continue;
            }
            outputs.push(SourceAdjacentAlignedPartial {
                partition_index,
                partition_count,
                key_columns: key_columns.to_vec(),
                partial: partition,
            });
        }
    }
    Ok(outputs)
}

fn execute_source_adjacent_raw_source_batch(
    execution: &mut BoundedExecutionAttempt<'_>,
    bounded: &BoundedPlan<'_>,
    batch: SourceChunkBatch,
) -> Result<Vec<(Vec<SourceChunk>, RawSourceInput)>> {
    execution.context().check_cancellation("query_engine.exec.bounded.source_adjacent_raw_source_batch_retry")?;
    if batch.is_singleton() {
        let chunk = batch.chunks.into_iter().next().expect("singleton source batch has one child");
        let raw = execute_bounded_raw_source_chunk(execution, bounded, &chunk)?;
        return Ok(vec![(vec![chunk], raw)]);
    }
    match execute_bounded_raw_source_chunk_batch(execution, bounded, &batch) {
        Ok(raw) => Ok(vec![(batch.chunks, raw)]),
        Err(error) if is_retryable_chunk_error(&error) => {
            record_chunk_retry_error(&error, execution.metrics());
            execution.metrics().record_source_chunk_retry();
            let Some((left, right)) = batch.split_for_retry() else {
                return Err(error);
            };
            backoff_before_retry(
                execution.context(),
                "query_engine.exec.bounded.source_adjacent_raw_source_batch_split_retry",
            )?;
            let mut raw_inputs = execute_source_adjacent_raw_source_batch(execution, bounded, left)?;
            match execute_source_adjacent_raw_source_batch(execution, bounded, right) {
                Ok(mut right_inputs) => {
                    raw_inputs.append(&mut right_inputs);
                    Ok(raw_inputs)
                },
                Err(error) => {
                    release_raw_source_inputs_reverse(raw_inputs);
                    Err(error)
                },
            }
        },
        Err(error) => Err(error),
    }
}

fn release_raw_source_inputs_reverse(mut inputs: Vec<(Vec<SourceChunk>, RawSourceInput)>) {
    while inputs.pop().is_some() {}
}

/// Plans the next raw source batch from the remaining chunks.
///
/// Row caps come from working-set feedback; the encoded-byte target remains
/// the policy ceiling. Sources with gates that forbid multi-chunk reads keep
/// singleton batches.
fn source_adjacent_next_raw_source_batch(
    memory_policy: MemoryControlPolicy,
    bounded: &BoundedPlan<'_>,
    remaining_chunks: &[SourceChunk],
    max_batch_rows: Option<u64>,
) -> SourceChunkBatch {
    let mut batches = source_adjacent_raw_source_batches(memory_policy, bounded, remaining_chunks, max_batch_rows);
    batches.truncate(1);
    batches.pop().expect("non-empty remaining chunks plan at least one batch")
}

fn source_adjacent_raw_source_batches(
    memory_policy: MemoryControlPolicy,
    bounded: &BoundedPlan<'_>,
    chunks: &[SourceChunk],
    max_batch_rows: Option<u64>,
) -> Vec<SourceChunkBatch> {
    if !bounded.source.partition_columns.is_empty()
        || bounded.source.files.iter().any(LocalParquetFile::has_row_delete_files)
        || bounded.source.scan_limit.is_some()
        || bounded.source.reader_predicate.as_ref().is_some_and(|predicate| {
            !bounded.source.reader_predicate_is_supported(predicate, rapids_analysis_context())
        })
    {
        return singleton_source_chunk_batches(chunks);
    }
    // Grouped aggregates hash-partition each raw batch by group key, so one
    // low-cardinality key concentrates a whole batch into a single partition
    // that is retained while every partition slices. Cap grouped raws at the
    // grouped target instead of the general cap so the worst-case partition
    // stays sliceable inside the attempt grant.
    let grouped = is_source_adjacent_grouped_aggregate(bounded);
    let max_batch_source_read_bytes = BoundedSourceBatchingPolicy::default()
        .source_adjacent_raw_source_batch_target_bytes(memory_policy, false, grouped);
    plan_source_chunk_batches_with_row_cap(chunks, max_batch_source_read_bytes, max_batch_rows)
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
            source_batching_reason: "source_gate:source_adjacent_batching_disabled".to_string(),
        })
        .collect()
}

fn source_adjacent_partitioned_aggregate_input_evidence(
    chunks: &[SourceChunk],
    source_row_group_count: u64,
    partitioner: &SourceAdjacentAggregatePartitioner,
    partition_count: usize,
    non_empty_partition_count: usize,
    max_partition_fan_in: usize,
) -> PartialAggregateInputEvidence {
    PartialAggregateInputEvidence {
        source_chunk_fingerprint: partial_aggregate_source_chunk_fingerprint(chunks),
        source_adjacent_partial: true,
        source_chunk_row_group_count: source_row_group_count,
        source_chunk_reads_all_row_groups: chunks.iter().all(|chunk| chunk.reads_all_row_groups),
        source_chunk_known_row_count: sum_optional_u64(chunks.iter().map(|chunk| chunk.known_row_count)),
        source_chunk_estimated_source_read_bytes: sum_optional_u64(
            chunks.iter().map(|chunk| chunk.estimated_source_read_bytes),
        ),
        source_chunk_size_estimate_based: chunks.iter().any(|chunk| chunk.size_estimate_based),
        source_adjacent_partition_strategy: partitioner.label(),
        source_adjacent_partition_count: Some(partition_count as u64),
        source_adjacent_non_empty_partition_count: Some(non_empty_partition_count as u64),
        source_adjacent_partition_key_count: partitioner.key_count(),
        source_adjacent_max_partition_fan_in: Some(max_partition_fan_in as u64),
    }
}

fn source_adjacent_partition_batch_ready(context: &ExecutionContext, inputs: &[SourceAdjacentPartitionInput]) -> bool {
    BoundedSourceBatchingPolicy::default().source_adjacent_partition_batch_ready(
        inputs.len(),
        inputs.iter().map(|input| input.source_row_group_count).sum(),
        source_adjacent_partition_batch_reserved_bytes(inputs),
        bounded_device_budget_bytes(context),
    )
}

fn source_adjacent_partition_batch_reserved_bytes(inputs: &[SourceAdjacentPartitionInput]) -> u64 {
    if inputs.len() <= 1 {
        return 0;
    }
    inputs
        .iter()
        .map(|input| input.partial.device_bytes())
        .try_fold(0u64, |total, bytes| total.checked_add(bytes))
        .unwrap_or(u64::MAX)
}

fn bounded_device_budget_bytes(context: &ExecutionContext) -> Option<u64> {
    match context.memory_control_policy() {
        MemoryControlPolicy::Bounded { device_budget_bytes, .. } => Some(device_budget_bytes),
        _ => None,
    }
}

#[cfg(test)]
fn source_adjacent_partition_batch_byte_limit_for_budget(device_budget_bytes: Option<u64>) -> u64 {
    BoundedSourceBatchingPolicy::default().source_adjacent_partition_batch_byte_limit_for_budget(device_budget_bytes)
}

fn execute_source_adjacent_partition_aggregate_batch(
    execution: &mut BoundedExecutionAttempt<'_>,
    bounded: &BoundedPlan<'_>,
    partition_index: usize,
    partition_count: usize,
    partitioner: &SourceAdjacentAggregatePartitioner,
    inputs: &mut Vec<SourceAdjacentPartitionInput>,
) -> Result<Vec<SourceAdjacentAlignedPartial>> {
    let batch_inputs = std::mem::take(inputs);
    let non_empty_partition_count = usize::from(batch_inputs.iter().any(|input| input.partial.frame.num_rows() > 0));
    let max_partition_fan_in = batch_inputs.len();
    let source_row_group_count = batch_inputs.iter().map(|input| input.source_row_group_count).sum::<u64>();
    let batch_chunks = batch_inputs.iter().flat_map(|input| input.chunks.iter().cloned()).collect::<Vec<_>>();
    let input_evidence = source_adjacent_partitioned_aggregate_input_evidence(
        &batch_chunks,
        source_row_group_count,
        partitioner,
        partition_count,
        non_empty_partition_count,
        max_partition_fan_in,
    );
    let partials = batch_inputs.into_iter().map(|input| input.partial).collect::<Vec<_>>();
    let partition_input = concatenate_partition_inputs(execution, bounded.aggregate_id, partition_index, partials)?;
    let partials = execute_source_adjacent_partition_aggregate_with_retry(
        execution,
        bounded,
        SOURCE_ADJACENT_PARTITIONED_PARTIAL_AGGREGATE_STAGE,
        partition_index,
        partition_input,
        input_evidence,
        source_adjacent_partition_aggregate_retry_budget(execution.context()),
    )?;
    Ok(partials
        .into_iter()
        .map(|partial| SourceAdjacentAlignedPartial {
            partition_index,
            partition_count,
            key_columns: partitioner.group_key_columns().unwrap_or(&[]).to_vec(),
            partial,
        })
        .collect())
}

fn source_adjacent_aggregate_partitioner(bounded: &BoundedPlan<'_>) -> SourceAdjacentAggregatePartitioner {
    let Ok(lowered_keys) = lower_aggregate_spec_keys(&bounded.aggregate.0.spec) else {
        return SourceAdjacentAggregatePartitioner::RowBalanced;
    };
    if lowered_keys.key_columns.is_empty() {
        return SourceAdjacentAggregatePartitioner::RowBalanced;
    }
    let mut has_string_key = false;
    for key_column in &lowered_keys.key_columns {
        let Ok(field) = bounded.aggregate_input_schema.field_with_name(key_column) else {
            return SourceAdjacentAggregatePartitioner::RowBalanced;
        };
        has_string_key |=
            matches!(field.data_type(), ArrowDataType::Utf8 | ArrowDataType::LargeUtf8 | ArrowDataType::Utf8View);
    }
    if !has_string_key {
        return SourceAdjacentAggregatePartitioner::RowBalanced;
    }
    // Variable-width keys always hash-partition by group key. Chunk-count no longer
    // demotes the path to a row-balanced streaming partition.
    SourceAdjacentAggregatePartitioner::GroupKey { key_columns: lowered_keys.key_columns }
}

/// Hash-partitioned chunk handed from the partition step to the slice loop.
///
/// Carrying the partitioned frame out by value ends the raw input's lifetime at
/// the partition boundary: `hash_partition_by` has already copied every row into
/// an independent table, so keeping the input alive across the slice loop would
/// hold a whole decoded chunk beside the partitioned copy for no further use.
struct PartitionedSourceAdjacentChunk {
    frame: cudf::DataFrame,
    offsets: Vec<usize>,
    original_columns: Option<Vec<String>>,
    token: ExecutionBudgetToken,
}

fn partition_source_adjacent_raw_chunk(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    chunk_index: usize,
    raw: CommittedPartial,
    partition_count: usize,
    partitioner: &SourceAdjacentAggregatePartitioner,
) -> Result<Vec<Option<CommittedPartial>>> {
    let PartitionedSourceAdjacentChunk { frame: partitioned_frame, offsets, original_columns, token } =
        hash_partition_source_adjacent_raw_chunk(execution, node_id, chunk_index, raw, partition_count, partitioner)?;
    let partitioned_gpu = GpuDataFrame::new(partitioned_frame.clone());
    match observe_device_memory_bytes(&partitioned_gpu, execution.metrics()) {
        Ok(_) => {},
        Err(error) => {
            return Err(error);
        },
    }

    let mut partitions = Vec::with_capacity(partition_count);
    for partition in 0..partition_count {
        match slice_source_adjacent_aggregate_partition(
            execution,
            node_id,
            &partitioned_frame,
            &offsets,
            original_columns.as_deref(),
            partition,
            partitioner.retain_empty_partition(partition),
            &token,
        ) {
            Ok(frame) => partitions.push(frame),
            Err(error) => {
                release_optional_partials_reverse(partitions);
                return Err(error);
            },
        }
    }
    Ok(partitions)
}

fn hash_partition_source_adjacent_raw_chunk(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    chunk_index: usize,
    raw: CommittedPartial,
    partition_count: usize,
    partitioner: &SourceAdjacentAggregatePartitioner,
) -> Result<PartitionedSourceAdjacentChunk> {
    let token = match execution_budget_token_for_boundary(
        execution.context(),
        execution.device_budget_bytes(),
        "source_adjacent_aggregate_hash_partition",
        "query_engine.bounded.aggregate.source_adjacent_partitioned_hash.budget_token",
        "failed to create source-adjacent partitioned aggregate budget token",
    ) {
        Ok(token) => token,
        Err(error) => {
            return Err(error);
        },
    };

    let raw_frame = raw.frame.into_frame();
    let original_columns = matches!(partitioner, SourceAdjacentAggregatePartitioner::RowBalanced)
        .then(|| raw_frame.column_names().into_iter().map(str::to_string).collect::<Vec<_>>());
    let input_rows = raw_frame.num_rows() as u64;
    let (observation, plan_execution_id, parent_span_id) = stage_context_from_execution(execution.context());
    let stage_guard = RuntimeOperatorStageGuard::start(
        observation,
        plan_execution_id,
        parent_span_id,
        RuntimeOperatorStageKind::HashPartition,
        RuntimeOperatorStageIdentity::new()
            .with_node_id(node_id.to_string())
            .with_chunk_index(chunk_index)
            .with_input_rows(input_rows),
    );
    let partitioned = match partitioner {
        SourceAdjacentAggregatePartitioner::RowBalanced => {
            let partition_column = unique_source_adjacent_partition_column_name(&raw_frame);
            let row_count = raw_frame.num_rows();
            let with_partition_column = raw_frame
                .with_execution(&token.execution_overrides())
                .with_position_column(&partition_column)
                .map_err(|error| {
                    attach_budget_token_failure_facts(
                        cudf_query_error(
                            "query_engine.bounded.aggregate.source_adjacent_partition_column",
                            "failed to add row-balanced source-adjacent aggregate partition column",
                            error,
                        )
                        .with_fact("node_id", node_id)
                        .with_fact("partition_count", partition_count)
                        .with_fact("input_rows", row_count),
                        &token,
                    )
                });
            let with_partition_column = match with_partition_column {
                Ok(frame) => frame,
                Err(error) => {
                    stage_guard.fail(&error);
                    return Err(error);
                },
            };
            with_partition_column
                .with_execution(&token.execution_overrides())
                .hash_partition_by([partition_column.as_str()], partition_count)
                .map_err(|error| {
                    attach_budget_token_failure_facts(
                        cudf_query_error(
                            "query_engine.bounded.aggregate.source_adjacent_hash_partition",
                            "failed to hash-partition source-adjacent aggregate input",
                            error,
                        )
                        .with_fact("node_id", node_id)
                        .with_fact("partition_key", partition_column.as_str())
                        .with_fact("partition_count", partition_count),
                        &token,
                    )
                })
        },
        SourceAdjacentAggregatePartitioner::GroupKey { key_columns } => raw_frame
            .with_execution(&token.execution_overrides())
            .hash_partition_by(key_columns.iter().map(String::as_str), partition_count)
            .map_err(|error| {
                attach_budget_token_failure_facts(
                    cudf_query_error(
                        "query_engine.bounded.aggregate.source_adjacent_group_key_hash_partition",
                        "failed to hash-partition source-adjacent aggregate input by group keys",
                        error,
                    )
                    .with_fact("node_id", node_id)
                    .with_fact("partition_keys", key_columns.join(","))
                    .with_fact("partition_count", partition_count),
                    &token,
                )
            }),
    };
    let partitioned = match partitioned {
        Ok(partitioned) => {
            stage_guard.finish(Some(input_rows), None);
            Ok(partitioned)
        },
        Err(error) => {
            stage_guard.fail(&error);
            Err(error)
        },
    };

    let partitioned = match partitioned {
        Ok(partitioned) => partitioned,
        Err(error) => {
            return Err(error);
        },
    };
    let (partitioned_frame, offsets) = partitioned.into_parts();
    Ok(PartitionedSourceAdjacentChunk { frame: partitioned_frame, offsets, original_columns, token })
}

#[expect(
    clippy::too_many_arguments,
    reason = "partition slice inputs (frame, offsets, output columns, partition identity) and its execution budget are distinct domain values"
)]
fn slice_source_adjacent_aggregate_partition(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    frame: &cudf::DataFrame,
    offsets: &[usize],
    output_columns: Option<&[String]>,
    partition: usize,
    retain_empty: bool,
    token: &ExecutionBudgetToken,
) -> Result<Option<CommittedPartial>> {
    let start = *offsets.get(partition).ok_or_else(|| {
        unsupported_bounded_shape(
            "source-adjacent aggregate partition is missing a partition start offset",
            "partition_offset_missing",
        )
    })?;
    let end = *offsets.get(partition + 1).ok_or_else(|| {
        unsupported_bounded_shape(
            "source-adjacent aggregate partition is missing a partition end offset",
            "partition_offset_missing",
        )
    })?;
    let len = end.saturating_sub(start);
    if len == 0 && !retain_empty {
        return Ok(None);
    }
    let sliced = budgeted_value(
        frame.slice_with_budget(start, len, token).map_err(|error| {
            cudf_query_error_with_budget_details(
                "query_engine.bounded.aggregate.source_adjacent_hash_partition_slice",
                "failed to slice source-adjacent hash-partitioned aggregate input",
                error,
            )
            .with_fact("node_id", node_id)
            .with_fact("partition", partition)
            .with_fact("partition_rows", len)
        })?,
        execution.metrics(),
        "query_engine.bounded.aggregate.source_adjacent_hash_partition_slice",
    );
    let sliced = if let Some(output_columns) = output_columns {
        let output_refs = output_columns.iter().map(String::as_str).collect::<Vec<_>>();
        sliced.select(output_refs).map_err(|error| {
            cudf_query_error(
                "query_engine.bounded.aggregate.source_adjacent_hash_partition_select",
                "failed to remove source-adjacent hash-partition helper column",
                error,
            )
            .with_fact("node_id", node_id)
            .with_fact("partition", partition)
            .with_fact("partition_rows", len)
        })?
    } else {
        sliced
    };
    let gpu = GpuDataFrame::new(sliced);
    let bytes = observe_device_memory_bytes(&gpu, execution.metrics())?;
    Ok(Some(ResidentGpuFrame::with_device_bytes(gpu, bytes)))
}

fn concatenate_partition_inputs(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    partition_index: usize,
    mut inputs: Vec<CommittedPartial>,
) -> Result<CommittedPartial> {
    if inputs.len() == 1 {
        return Ok(inputs.pop().expect("single input checked"));
    }
    concatenate_committed_frames(
        execution,
        node_id,
        partition_index,
        inputs,
        ReservationKind::OperatorOutput,
        "source_adjacent_aggregate_partition_concat",
        "source_adjacent_aggregate_partition_concat_output",
        "query_engine.bounded.aggregate.source_adjacent_partition_concat.budget_token",
        "failed to create source-adjacent aggregate partition concat budget token",
        "query_engine.bounded.aggregate.source_adjacent_partition_concat",
        "failed to concatenate source-adjacent aggregate partition inputs",
    )
}

const SOURCE_ADJACENT_PARTITIONED_PARTIAL_AGGREGATE_STAGE: &str = "source_adjacent_partitioned_partial_aggregate";

/// A source-adjacent partition may be split at rows only after source-local
/// stages have completed. Each half is still a valid partial-aggregate input,
/// and the existing final merge owns combining its partial states.
fn execute_source_adjacent_partition_aggregate_with_retry(
    execution: &mut BoundedExecutionAttempt<'_>,
    bounded: &BoundedPlan<'_>,
    stage: &'static str,
    partition_index: usize,
    input: CommittedPartial,
    input_evidence: PartialAggregateInputEvidence,
    retry_budget: usize,
) -> Result<Vec<CommittedPartial>> {
    let upper_bound_bytes = match partial_aggregate_work_reservation_upper_bound_bytes(
        execution,
        bounded.aggregate.0,
        bounded.aggregate_input_schema,
        bounded.aggregate.1,
        &input.frame,
    ) {
        Ok(bytes) => bytes,
        Err(error) => {
            return Err(source_adjacent_partition_aggregate_error(error, bounded.aggregate_id, stage, partition_index));
        },
    };
    if source_adjacent_partition_aggregate_should_split(
        input.frame.num_rows(),
        upper_bound_bytes,
        retry_budget,
        execution.device_budget_bytes(),
    ) {
        execution.metrics().record_retry_resource_exhausted();
        if let Err(error) = backoff_before_retry(
            execution.context(),
            "query_engine.exec.bounded.source_adjacent_partition_aggregate_split_retry",
        ) {
            return Err(source_adjacent_partition_aggregate_error(error, bounded.aggregate_id, stage, partition_index));
        }
        let (left, right) = split_source_adjacent_partition_aggregate_input(
            execution,
            bounded.aggregate_id,
            stage,
            partition_index,
            input,
        )?;
        let mut partials = match execute_source_adjacent_partition_aggregate_with_retry(
            execution,
            bounded,
            stage,
            partition_index,
            left,
            input_evidence,
            retry_budget.saturating_sub(1),
        ) {
            Ok(partials) => partials,
            Err(error) => {
                return Err(error);
            },
        };
        match execute_source_adjacent_partition_aggregate_with_retry(
            execution,
            bounded,
            stage,
            partition_index,
            right,
            input_evidence,
            retry_budget.saturating_sub(1),
        ) {
            Ok(mut right_partials) => {
                partials.append(&mut right_partials);
                Ok(partials)
            },
            Err(error) => {
                release_partials_reverse(partials);
                Err(error)
            },
        }
    } else {
        execute_source_adjacent_partition_aggregate_with_upper_bound(
            execution,
            bounded,
            stage,
            partition_index,
            input,
            input_evidence,
            upper_bound_bytes,
        )
        .map(|partial| vec![partial])
    }
}

fn execute_source_adjacent_partition_aggregate_with_upper_bound(
    execution: &mut BoundedExecutionAttempt<'_>,
    bounded: &BoundedPlan<'_>,
    stage: &'static str,
    partition_index: usize,
    input: CommittedPartial,
    input_evidence: PartialAggregateInputEvidence,
    upper_bound_bytes: u64,
) -> Result<CommittedPartial> {
    let input_rows = input.frame.num_rows() as u64;
    let input_frame = input.frame;
    let task_id = source_adjacent_aggregate_task_id(bounded.aggregate_id, stage, format!("partition{partition_index}"));
    let _ = upper_bound_bytes;
    let token = match execution_budget_token_for_boundary(
        execution.context(),
        execution.device_budget_bytes(),
        stage,
        "query_engine.bounded.aggregate.source_adjacent_partial.budget_token",
        "failed to create source-adjacent partial aggregate budget token",
    ) {
        Ok(token) => token,
        Err(error) => {
            return Err(source_adjacent_partition_aggregate_error(error, bounded.aggregate_id, stage, partition_index));
        },
    };
    let chunk_operator_timer = Instant::now();
    let budget_window = execution.metrics().open_budget_window();
    let result = execute_partial_aggregate_with_budget(
        execution.context(),
        Some(bounded.aggregate_id),
        bounded.aggregate.0,
        bounded.aggregate_input_schema,
        bounded.aggregate.1,
        input_frame,
        &token,
        Some(input_evidence),
        execution.metrics(),
    );
    execution.metrics().close_budget_window(&task_id, budget_window);
    execution.metrics().add_chunk_operator_time(chunk_operator_timer.elapsed());
    let frame = match result {
        Ok(frame) => frame,
        Err(error) => {
            return Err(source_adjacent_partition_aggregate_error(error, bounded.aggregate_id, stage, partition_index));
        },
    };
    let output_bytes = match observe_device_memory_bytes(&frame, execution.metrics()) {
        Ok(bytes) => bytes,
        Err(error) => {
            return Err(source_adjacent_partition_aggregate_error(error, bounded.aggregate_id, stage, partition_index));
        },
    };
    let output = ResidentGpuFrame::with_device_bytes(frame, output_bytes);
    execution.metrics().record_partial_aggregate(input_rows, output.frame.num_rows() as u64, output.device_bytes());
    Ok(output)
}

fn source_adjacent_partition_aggregate_retry_budget(context: &ExecutionContext) -> usize {
    match context.memory_control_policy() {
        MemoryControlPolicy::Bounded { max_retry_attempts, .. } => max_retry_attempts.saturating_sub(1),
        MemoryControlPolicy::Unbounded => 0,
    }
}

fn source_adjacent_partition_aggregate_should_split(
    input_rows: usize,
    upper_bound_bytes: u64,
    retry_budget: usize,
    remaining_bytes: u64,
) -> bool {
    retry_budget > 0 && input_rows > 1 && upper_bound_bytes > remaining_bytes
}

fn split_source_adjacent_partition_aggregate_input(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    stage: &'static str,
    partition_index: usize,
    input: CommittedPartial,
) -> Result<(CommittedPartial, CommittedPartial)> {
    let input_rows = input.frame.num_rows();
    let split_at = input_rows / 2;
    if split_at == 0 || split_at == input_rows {
        return Err(source_adjacent_partition_aggregate_error(
            unsupported_bounded_shape(
                "source-adjacent partitioned aggregate input cannot be split further",
                "source_adjacent_partitioned_partial_aggregate_retry_unsplittable",
            )
            .with_fact("input_rows", input_rows),
            node_id,
            stage,
            partition_index,
        ));
    }
    let source_frame = input.frame.into_frame();
    let overrides_lease = match execution_overrides(execution.context()) {
        Ok(lease) => lease,
        Err(error) => {
            return Err(source_adjacent_partition_aggregate_error(error, node_id, stage, partition_index));
        },
    };
    let overrides = overrides_lease.overrides();
    let left = match source_frame.with_execution(&overrides).slice(0, split_at) {
        Ok(frame) => frame,
        Err(error) => {
            return Err(source_adjacent_partition_aggregate_error(
                cudf_query_error(
                    "query_engine.bounded.aggregate.source_adjacent_partition_split",
                    "failed to split source-adjacent partitioned aggregate input",
                    error,
                )
                .with_fact("split_side", "left")
                .with_fact("input_rows", input_rows)
                .with_fact("split_at", split_at),
                node_id,
                stage,
                partition_index,
            ));
        },
    };
    let right_len = input_rows.saturating_sub(split_at);
    let right = match source_frame.with_execution(&overrides).slice(split_at, right_len) {
        Ok(frame) => frame,
        Err(error) => {
            return Err(source_adjacent_partition_aggregate_error(
                cudf_query_error(
                    "query_engine.bounded.aggregate.source_adjacent_partition_split",
                    "failed to split source-adjacent partitioned aggregate input",
                    error,
                )
                .with_fact("split_side", "right")
                .with_fact("input_rows", input_rows)
                .with_fact("split_at", split_at),
                node_id,
                stage,
                partition_index,
            ));
        },
    };
    let left = GpuDataFrame::new(left);
    let right = GpuDataFrame::new(right);
    let left_bytes = match observe_device_memory_bytes(&left, execution.metrics()) {
        Ok(bytes) => bytes,
        Err(error) => return Err(source_adjacent_partition_aggregate_error(error, node_id, stage, partition_index)),
    };
    let right_bytes = match observe_device_memory_bytes(&right, execution.metrics()) {
        Ok(bytes) => bytes,
        Err(error) => return Err(source_adjacent_partition_aggregate_error(error, node_id, stage, partition_index)),
    };
    Ok((ResidentGpuFrame::with_device_bytes(left, left_bytes), ResidentGpuFrame::with_device_bytes(right, right_bytes)))
}

fn source_adjacent_partition_aggregate_error(
    error: QueryError,
    node_id: NodeId,
    stage: &'static str,
    partition_index: usize,
) -> QueryError {
    error
        .with_fact_if_absent("node_id", node_id)
        .with_fact_if_absent("partition", partition_index)
        .with_fact_if_absent("stage", stage)
}

#[expect(
    clippy::too_many_arguments,
    reason = "bounded execution passes operator-occurrence inputs explicitly beside the attempt state"
)]
fn concatenate_committed_frames(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    partition_index: usize,
    mut inputs: Vec<CommittedPartial>,
    reservation_kind: ReservationKind,
    stage: &'static str,
    output_boundary: &'static str,
    token_operation: &'static str,
    token_message: &'static str,
    concat_operation: &'static str,
    concat_message: &'static str,
) -> Result<CommittedPartial> {
    if inputs.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            concat_operation,
            "source-adjacent partitioned aggregate concat requires at least one input frame",
        )
        .with_fact("node_id", node_id)
        .with_fact("partition", partition_index));
    }
    let token = match execution_budget_token_for_boundary(
        execution.context(),
        execution.device_budget_bytes(),
        stage,
        token_operation,
        token_message,
    ) {
        Ok(token) => token,
        Err(error) => {
            release_partials_reverse(inputs);
            return Err(error);
        },
    };

    let first = inputs.remove(0);
    let first_frame = first.frame.into_frame();
    let concatenated = {
        let other_frames = inputs.iter().map(|input| input.frame.frame().clone()).collect::<Vec<_>>();
        let other_refs = other_frames.iter().collect::<Vec<_>>();
        first_frame.with_execution(&token.execution_overrides()).concat_many(&other_refs).map_err(|error| {
            attach_budget_token_failure_facts(
                cudf_query_error(concat_operation, concat_message, error)
                    .with_fact("node_id", node_id)
                    .with_fact("partition", partition_index)
                    .with_fact("input_count", inputs.len() + 1),
                &token,
            )
        })
    };
    release_partials_reverse(inputs);

    let concatenated = match concatenated {
        Ok(frame) => GpuDataFrame::new(frame),
        Err(error) => {
            return Err(error);
        },
    };
    let output_bytes = match observe_device_memory_bytes(&concatenated, execution.metrics()) {
        Ok(bytes) => bytes,
        Err(error) => {
            return Err(error);
        },
    };
    let _ = (reservation_kind, output_boundary);
    Ok(ResidentGpuFrame::with_device_bytes(concatenated, output_bytes))
}

fn release_nested_partition_inputs_reverse(mut partitions: Vec<Vec<SourceAdjacentPartitionInput>>) {
    while let Some(partition) = partitions.pop() {
        release_partition_inputs_reverse(partition);
    }
}

fn release_partition_inputs_reverse(mut inputs: Vec<SourceAdjacentPartitionInput>) {
    while inputs.pop().is_some() {}
}

fn release_optional_partials_reverse(mut partials: Vec<Option<CommittedPartial>>) {
    while partials.pop().is_some() {}
}

fn release_aligned_partials_reverse(mut partials: Vec<SourceAdjacentAlignedPartial>) {
    while partials.pop().is_some() {}
}

fn unique_source_adjacent_partition_column_name(frame: &cudf::DataFrame) -> String {
    let base = "__algeon_native_source_adjacent_partition_row_id";
    unique_alias_with_suffix(base, |candidate| frame.contains(candidate))
}

fn source_adjacent_aggregate_task_id(node_id: NodeId, stage: &'static str, suffix: impl std::fmt::Display) -> String {
    format!("aggregate.node{node_id}.{stage}.{suffix}")
}

fn source_adjacent_partitioned_aggregate_min_partition_count(chunk_count: usize) -> usize {
    BoundedPartitionRowPolicy::default().source_adjacent_partitioned_aggregate_min_partition_count(chunk_count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_adjacent_partition_batch_byte_limit_tracks_device_budget_with_bounds() {
        assert_eq!(
            source_adjacent_partition_batch_byte_limit_for_budget(Some(16 * 1024 * 1024 * 1024)),
            BoundedSourceBatchingPolicy::DEFAULT_SOURCE_ADJACENT_PARTITIONED_AGGREGATE_MAX_BATCH_BYTES
        );
        assert_eq!(
            source_adjacent_partition_batch_byte_limit_for_budget(Some(4 * 1024 * 1024 * 1024)),
            512 * 1024 * 1024
        );
        assert_eq!(
            source_adjacent_partition_batch_byte_limit_for_budget(Some(256 * 1024 * 1024)),
            BoundedSourceBatchingPolicy::DEFAULT_SOURCE_ADJACENT_PARTITIONED_AGGREGATE_MIN_BATCH_BYTES
        );
        assert_eq!(
            source_adjacent_partition_batch_byte_limit_for_budget(None),
            BoundedSourceBatchingPolicy::DEFAULT_SOURCE_ADJACENT_PARTITIONED_AGGREGATE_MAX_BATCH_BYTES
        );
    }

    #[test]
    fn source_adjacent_partial_aggregate_split_requires_budget_pressure_and_a_safe_retry() {
        assert!(!source_adjacent_partition_aggregate_should_split(1, 101, 1, 100));
        assert!(!source_adjacent_partition_aggregate_should_split(2, 101, 0, 100));
        assert!(!source_adjacent_partition_aggregate_should_split(2, 100, 1, 100));
        assert!(source_adjacent_partition_aggregate_should_split(2, 101, 1, 100));
    }

    #[test]
    fn grouped_aggregate_raw_batches_respect_grouped_target() {
        use std::sync::Arc;

        use arrow::datatypes::{DataType, Field, Schema, SchemaRef};

        use crate::{
            expr::{AggregateSpec, Expr, LiteralValue},
            plan::{AggregateNode, FilterNode, NodeId},
            runtime::SourceChunkPlanningMode,
            source::{LocalParquetFile, LocalParquetSource, ParquetSourceLocation, ResolvedParquetSource},
        };

        const CHUNK_BYTES: u64 = 300 * 1024 * 1024;
        const GROUPED_TARGET_BYTES: u64 =
            BoundedSourceBatchingPolicy::DEFAULT_SOURCE_ADJACENT_GROUPED_AGGREGATE_TARGET_CHUNK_BYTES;

        fn batchable_chunk(index: usize, bytes: u64) -> SourceChunk {
            SourceChunk {
                source_node_id: NodeId::from_index_for_internal_use(1),
                chunk_index: index,
                file_index: 0,
                location: ParquetSourceLocation::LocalPath(std::path::PathBuf::from("part-0.parquet")),
                path: std::path::PathBuf::from("part-0.parquet"),
                resolved_source: ResolvedParquetSource::local_kvikio("part-0.parquet"),
                stable_id: "file-0".to_string(),
                row_group_indexes: vec![index],
                reads_all_row_groups: false,
                known_file_bytes: Some(bytes),
                known_row_group_bytes: Some(bytes),
                known_projected_compressed_bytes: Some(bytes),
                known_projected_uncompressed_bytes: Some(bytes),
                known_pure_filter_compressed_bytes: Some(0),
                known_pure_filter_uncompressed_bytes: Some(0),
                estimated_chunk_bytes: Some(bytes),
                estimated_source_read_bytes: Some(bytes),
                exact_source_read_preflight_bytes: Some(bytes),
                source_file_row_groups: None,
                source_file_row_group_bytes: None,
                source_file_projected_compressed_bytes: None,
                source_file_projected_uncompressed_bytes: None,
                source_file_pure_filter_compressed_bytes: None,
                source_file_pure_filter_uncompressed_bytes: None,
                source_file_row_group_rows: None,
                reader_decimal_types: Vec::new(),
                reader_decimal_types_footer_verified: false,
                read_columns: vec!["k".to_string()],
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

        fn plan_with_group_exprs<'a>(
            source: &'a LocalParquetSource,
            filter: &'a FilterNode,
            aggregate: &'a AggregateNode,
            schema: &'a SchemaRef,
        ) -> BoundedPlan<'a> {
            BoundedPlan {
                source_id: NodeId::from_index_for_internal_use(1),
                source,
                pre_filter_projections: Vec::new(),
                filter: Some(filter),
                post_filter_projections: Vec::new(),
                aggregate_id: NodeId::from_index_for_internal_use(3),
                aggregate_input_schema: schema,
                aggregate: (aggregate, schema),
                post_aggregate_stages: Vec::new(),
                sort: None,
                post_sort_projections: Vec::new(),
                limit: None,
            }
        }

        let schema: SchemaRef = Arc::new(Schema::new(vec![Field::new("k", DataType::Utf8, true)]));
        let source = LocalParquetSource::from_files(
            vec![LocalParquetFile::new("/tmp/grouped.parquet")],
            Arc::clone(&schema),
            vec!["k".to_string()],
        );
        let filter = FilterNode {
            input: NodeId::from_index_for_internal_use(1),
            predicate: Expr::Literal(LiteralValue::Boolean(true)),
        };
        let grouped = AggregateNode {
            input: NodeId::from_index_for_internal_use(2),
            spec: AggregateSpec::new(vec![Expr::Literal(LiteralValue::Utf8("k".to_string()))], Vec::new()),
        };
        let ungrouped = AggregateNode {
            input: NodeId::from_index_for_internal_use(2),
            spec: AggregateSpec::new(Vec::new(), Vec::new()),
        };
        let policy = MemoryControlPolicy::Bounded {
            device_budget_bytes: 94 * 1024 * 1024 * 1024,
            max_source_chunk_bytes: 32 * 1024 * 1024 * 1024,
            max_row_groups_per_chunk: 1024,
            max_retry_attempts: 4,
            min_retry_chunk_row_groups: 1,
        };
        let chunks: Vec<SourceChunk> = (0..8).map(|index| batchable_chunk(index, CHUNK_BYTES)).collect();

        // Grouped aggregates hash-partition each raw batch by group key, so a
        // low-cardinality key concentrates a whole batch into one partition.
        // Batches must stay at the grouped target even when the configured
        // chunk cap is much larger.
        let grouped_batches = source_adjacent_raw_source_batches(
            policy,
            &plan_with_group_exprs(&source, &filter, &grouped, &schema),
            &chunks,
            None,
        );
        assert_eq!(grouped_batches.len(), chunks.len());
        for batch in &grouped_batches {
            assert!(batch.is_singleton());
            assert!(
                batch.estimated_source_read_bytes.is_some_and(|bytes| bytes <= GROUPED_TARGET_BYTES),
                "grouped raw batch exceeds grouped target: {batch:?}"
            );
        }

        // The general cap still applies away from grouped aggregates.
        let ungrouped_batches = source_adjacent_raw_source_batches(
            policy,
            &plan_with_group_exprs(&source, &filter, &ungrouped, &schema),
            &chunks,
            None,
        );
        assert_eq!(ungrouped_batches.len(), 2);

        // A row cap from working-set feedback bounds a batch below the encoded
        // byte target, and a chunk without a known row count stays singleton.
        let row_capped_batches = source_adjacent_raw_source_batches(
            policy,
            &plan_with_group_exprs(&source, &filter, &ungrouped, &schema),
            &chunks,
            Some(3),
        );
        assert_eq!(row_capped_batches.iter().map(SourceChunkBatch::chunk_count).collect::<Vec<_>>(), vec![3, 3, 2]);
        let mut unknown_rows = chunks;
        unknown_rows[1].known_row_count = None;
        let unknown_row_batches = source_adjacent_raw_source_batches(
            policy,
            &plan_with_group_exprs(&source, &filter, &ungrouped, &schema),
            &unknown_rows,
            Some(3),
        );
        assert_eq!(unknown_row_batches.iter().map(SourceChunkBatch::chunk_count).collect::<Vec<_>>(), vec![1, 1, 3, 3]);
    }
}
