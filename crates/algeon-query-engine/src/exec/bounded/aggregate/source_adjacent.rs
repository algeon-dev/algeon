//! Grouped and global aggregates whose input is a local Parquet source, and the group-key
//! partitioned accumulation they share with grouped aggregates over resident frame streams.
//!
//! A source-adjacent aggregate (`BoundedPlan`) has only a filter and
//! projections between the source and the aggregate, so each source chunk can
//! be read, staged and partially aggregated without any other operator
//! holding a resident input. Two paths exist. The plain path commits one
//! partial per chunk batch into `CompiledStreamPartials` and lets the final
//! merge regroup them. The partitioned path, selected by
//! `should_execute_partitioned_source_adjacent_grouped_aggregate`, reads raw
//! batches, hash-partitions each by group key, retains the slices of every
//! partition until the batching policy releases them, aggregates each
//! partition's slices together, and feeds an incremental merge that keeps one
//! merged state per partition. Slices retained past the partition batch byte
//! limit, across all partitions, are a hold on the attempt's retained-state
//! ledger (`source_adjacent_hold_pending_partitions`); a refused hold
//! aggregates every partition at once. A partition that reaches neither bound
//! is aggregated once, from the slices of every batch, when the source ends.
//!
//! The partitioned path sizes its next raw batch from what the previous
//! batches decoded and staged (`SourceAdjacentRawBatchFeedback`), and from the
//! attempt's current allocation. The same feedback decides whether a batch is
//! reduced whole before partitioning. The first batch has no feedback, so each
//! of its raw inputs gathers a group-key sample of at most
//! `source_adjacent_local_reduction_key_sample_rows` rows and counts its
//! distinct keys. That probe runs once per first-batch raw input (again for
//! each piece of a retry split), and its cost is fixed by the sample bound,
//! not by the batch size. The partition count is fixed by the first raw input
//! and never changes within a run (`accumulation_partition_count`): the input
//! estimate, widened by how much the partial state of the first input's rows
//! can exceed those rows under the same byte model, divided by the aggregate
//! partition target. A partition holds its input and produces that state, so
//! the larger of the two sizes it; a group-only aggregate's state is a subset
//! of its input and leaves the estimate unchanged. With an `AggregateLaneDispatch`,
//! raw batches and ready partitions run as lane work and their results are
//! folded on the coordinator; without one everything runs inline. Retry is
//! local: a raw batch whose read, stages or partitioning fails on a resource
//! error splits by chunk, a single chunk splits by row group, and a partition
//! whose partial state is bounded above the cap splits by rows before it runs.
//!
//! The same accumulation serves a grouped aggregate over a resident frame
//! stream (`execute_partitioned_frame_stream_aggregate`). Each pulled frame is
//! a raw input (`PartitionedRawInput`) with the same staging decision,
//! readiness rules and holds; `PartitionedAggregate` names the aggregate in
//! place of a `BoundedPlan`. Source evidence, the batch row cap, lane splits
//! and the source retry apply to source batches alone. A frame has no source to
//! read again, so a resource failure while staging it ends the aggregate, as it
//! does for the per-frame partial path. The key sample decides from at most a
//! frame's rows; a frame too small for its sample to show reduction is
//! partitioned and held like a high-cardinality one, and the extra work that
//! costs is bounded by the frame.

use std::time::Instant;

use cudf::{DuplicateKeepOption, frame::dataframe::DropDuplicatesOptions};
use query_planning::{naming::unique_alias_with_suffix, util::sum_optional_u64};
use query_runtime::observability::{
    RuntimeOperatorStageGuard, RuntimeOperatorStageIdentity, RuntimeOperatorStageKind, stage_context_from_execution,
};

use super::{
    AggregateNode, Arc, ArrowDataType, ArrowSchemaRef, BoundedExecutionAttempt, BoundedPartitionRowPolicy, BoundedPlan,
    BoundedSourceBatchingPolicy, CommittedPartial, CompiledIncrementalPartitionedMerge, ExecutionBudgetToken,
    ExecutionContext, FilterNode, GpuDataFrame, LocalParquetSource, MemoryControlPolicy, NodeId,
    PartialAggregateInputEvidence, PlanningPolicy, ProjectionNode, QueryError, QueryErrorCode, ResidentGpuFrame,
    Result, SourceChunk, SourceChunkBatch, attach_budget_token_failure_facts, bounded_invariant_violation,
    budgeted_value, cudf_query_error, execute_bounded_raw_source_chunk, execute_bounded_raw_source_chunk_batch,
    execute_incremental_source_adjacent_final_aggregate_merge, execute_partial_aggregate_with_budget,
    execution_budget_token_for_boundary,
    final_merge::{first_batch_estimated_total_bytes, partition_final_merge_partial},
    is_retryable_chunk_error, is_source_adjacent_grouped_aggregate, lower_aggregate_spec_keys,
    lower_final_aggregate_spec, observe_device_memory_bytes, partial_aggregate_source_chunk_fingerprint,
    partial_aggregate_work_reservation_upper_bound_bytes,
    partitioned::partitioned_aggregate_partition_count,
    partitioned_accumulation_accepts, plan_source_chunk_batches_with_row_cap, record_chunk_retry_error,
    should_execute_incremental_partitioned_final_aggregate_merge,
    sizing::aggregate_input_upper_bound_bytes,
};
use crate::{
    exec::{
        bounded::{
            RetentionCharge,
            chunks::{
                bounded_source_adjacent_chunk_batches_for_work_width, bounded_source_adjacent_chunk_batching_enabled,
                chunk_retry_facts, execute_source_adjacent_batch_uncommitted, split_retryable_chunk,
            },
            compiled::{CompiledGpuResource, ResidentFrameStreamPayload, lanes::AggregateLaneDispatch},
        },
        retry::{RetryLimits, backoff_before_retry},
    },
    rapids_backend::rapids_analysis_context,
};

/// Source and aggregate ownership needed by a batch worker; downstream plan
/// stages remain with the compiled coordinator.
///
/// Lane jobs are `'static` pool closures, so they cannot borrow the
/// coordinator's `BoundedPlan`. One `Arc` of this clone is shared by every
/// lane job of a run and `plan()` rebuilds a borrowed plan view per batch.
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
            limit: None,
        }
    }
}

/// Runs a source-adjacent aggregate over `chunks` to its merged output.
///
/// `partitioned` selects the group-key partitioned path; otherwise chunks are
/// batched per the lane width and each batch yields one partial. A chunk's
/// source commit is recorded only after its partial has been accepted, so a
/// batch that fails leaves no committed chunk behind.
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

/// The aggregate a partitioned accumulation feeds: its node, spec and the schemas of its input
/// and output. Source batches take it from their `BoundedPlan`; resident frame streams from the
/// compiled aggregate task.
#[derive(Clone, Copy)]
struct PartitionedAggregate<'a> {
    node_id: NodeId,
    aggregate: &'a AggregateNode,
    input_schema: &'a ArrowSchemaRef,
    output_schema: &'a ArrowSchemaRef,
}

impl<'a> PartitionedAggregate<'a> {
    fn of_plan(bounded: &'a BoundedPlan<'_>) -> Self {
        Self {
            node_id: bounded.aggregate_id,
            aggregate: bounded.aggregate.0,
            input_schema: bounded.aggregate_input_schema,
            output_schema: bounded.aggregate.1,
        }
    }
}

/// Where one raw input of the partitioned accumulation came from. Only source inputs carry
/// source evidence; the key-sample offset seed and the stage's chunk index come from either.
#[derive(Clone)]
enum PartitionedInputOrigin {
    /// Staged source chunks, which are row-group halves of a planned chunk after a retry split,
    /// and the row groups the read covered.
    Source { chunks: Arc<[SourceChunk]>, row_group_count: u64 },
    /// The `index`-th frame pulled from a resident frame stream.
    Frame { index: usize },
}

impl PartitionedInputOrigin {
    /// Index recorded on the input's hash-partition and key-sample stages.
    fn chunk_index(&self) -> usize {
        match self {
            Self::Source { chunks, .. } => chunks.first().expect("raw source batch has at least one chunk").chunk_index,
            Self::Frame { index } => *index,
        }
    }

    /// Seed of the key sample's row offset, stable for one input.
    fn key_sample_seed(&self) -> u64 {
        match self {
            Self::Source { chunks, .. } => partial_aggregate_source_chunk_fingerprint(chunks).unwrap_or(0),
            Self::Frame { index } => *index as u64,
        }
    }

    fn source_row_group_count(&self) -> u64 {
        match self {
            Self::Source { row_group_count, .. } => *row_group_count,
            Self::Frame { .. } => 0,
        }
    }
}

/// One raw input of the partitioned accumulation: the frame the aggregate reads, with the rows
/// and decoded bytes it had before source-adjacent stages. A resident frame reports its own rows
/// and bytes, which only the source path's batch sizing reads.
struct PartitionedRawInput {
    origin: PartitionedInputOrigin,
    frame: CommittedPartial,
    rows: u64,
    decoded_bytes: u64,
}

/// One partition's slice of a raw input, resident until its partition is aggregated. `origin` is
/// kept for the input evidence of that aggregate.
struct SourceAdjacentPartitionInput {
    origin: PartitionedInputOrigin,
    partial: CommittedPartial,
}

/// Inputs of one raw batch worker.
///
/// `pending` carries the retained partition inputs, and `hold` their charge,
/// only on the inline path; lane workers receive `None` because the
/// coordinator keeps the retained inputs and folds each worker's new slices in
/// afterwards. `partition_count` is `None` only before the first batch has
/// fixed it, and `local_reduction` only before any batch has been observed.
struct PartitionedSourceBatchWork {
    batch: SourceChunkBatch,
    pending: Option<Vec<Vec<SourceAdjacentPartitionInput>>>,
    hold: Option<RetentionCharge>,
    partition_count: Option<usize>,
    local_reduction: Option<bool>,
    estimated_input_bytes: u64,
    min_partition_count: usize,
}

/// Outputs of one raw batch worker: partials ready for the final merge,
/// slices still pending per partition with the work's hold, one feedback sample per raw input, and
/// the (chunk count, row group count) commit the coordinator records once
/// the outputs are accepted.
struct PartitionedSourceBatchOutput {
    pending: Option<Vec<Vec<SourceAdjacentPartitionInput>>>,
    hold: Option<RetentionCharge>,
    partition_count: Option<usize>,
    outputs: Vec<SourceAdjacentAlignedPartial>,
    samples: Vec<SourceAdjacentRawBatchSample>,
    commit: (u64, u64),
}

/// Partial aggregate output that remains hash-aligned to a group-key partition.
struct SourceAdjacentAlignedPartial {
    partition_index: usize,
    partition_count: usize,
    key_columns: Vec<String>,
    partial: CommittedPartial,
}

/// Final merge sink of the partitioned path.
///
/// Group-key partitions are hash-aligned with the merge's own partitions, so
/// the aligned incremental merge pushes each partial straight into its
/// partition without re-hashing. Row-balanced partitions use the unaligned
/// incremental merge when the aggregate shape admits one, and otherwise fall
/// back to collecting partials for the compiled final merge.
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
                push_aligned_partials(execution, merge, outputs)
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

/// How a staged raw batch is split into partitions.
///
/// `GroupKey` hashes the group-key columns, so every row of one group lands in
/// the same partition and each partition can be aggregated to final groups
/// independently. `RowBalanced` hashes a synthetic row position instead, which
/// bounds each partition's size but leaves groups spread over partitions, so
/// its partials still need a cross-partition merge.
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

    /// Whether an empty partition slice is kept as a zero-row partial.
    /// Row-balanced runs keep every slice so the partition set has a fixed
    /// shape; group-key runs keep only partition 0, enough for an input with
    /// no rows to still reach the final merge with one empty partial.
    fn retain_empty_partition(&self, partition: usize) -> bool {
        match self {
            Self::RowBalanced => true,
            Self::GroupKey { .. } => partition == 0,
        }
    }
}

/// One completed raw source batch, as observed.
///
/// Decoded bytes come from the reader's output frame and staged bytes from the
/// frame after the source-adjacent filter and projections. The next batch uses
/// these observations to account for decoded working-set expansion. Partial rows
/// are known only when the batch produced partial output; partition occupancy is
/// known only when the batch was hash-partitioned.
#[derive(Debug, Default, Clone, Copy)]
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
    decoded_bytes_per_row: u64,
    staged_bytes_per_row: u64,
    last: Option<SourceAdjacentRawBatchSample>,
}

impl SourceAdjacentRawBatchFeedback {
    fn observe(&mut self, sample: SourceAdjacentRawBatchSample) {
        if sample.rows > 0 {
            self.decoded_bytes_per_row = self.decoded_bytes_per_row.max(sample.decoded_bytes.div_ceil(sample.rows));
            self.staged_bytes_per_row = self.staged_bytes_per_row.max(sample.staged_bytes.div_ceil(sample.rows));
        }
        self.last = Some(sample);
    }

    /// Row cap for the next raw batch, or `None` when neither an observation
    /// nor the planned chunks' decoded estimates can bound the working set.
    fn next_batch_row_cap(
        &self,
        policy: BoundedSourceBatchingPolicy,
        device_budget_bytes: u64,
        retained_device_bytes: u64,
        remaining_chunks: &[SourceChunk],
    ) -> Option<u64> {
        let target_bytes = policy.source_adjacent_raw_batch_working_set_target_bytes(device_budget_bytes).min(
            policy.source_adjacent_raw_batch_working_set_target_bytes(
                device_budget_bytes.saturating_sub(retained_device_bytes),
            ),
        );
        // Per-row high-water samples preserve wide batches across source files
        // and lane completion order while retained state reduces the next target.
        let bytes_per_row = if self.decoded_bytes_per_row > 0 {
            policy.source_adjacent_raw_batch_observed_working_set_bytes_per_row(
                self.decoded_bytes_per_row,
                self.staged_bytes_per_row,
            )
        } else {
            let (rows, decoded_bytes) = remaining_chunks
                .iter()
                .filter_map(|chunk| Some((chunk.known_row_count?, chunk.estimated_decoded_bytes?)))
                .fold((0u64, 0u64), |(rows, bytes), (chunk_rows, chunk_bytes)| {
                    (rows.saturating_add(chunk_rows), bytes.saturating_add(chunk_bytes))
                });
            if rows == 0 {
                return None;
            }
            policy.source_adjacent_raw_batch_cold_working_set_bytes_per_row(decoded_bytes.div_ceil(rows))
        };
        Some((target_bytes / bytes_per_row).max(1))
    }

    /// Whether the next batch should be reduced whole before partitioning, or
    /// `None` before any batch has been observed.
    ///
    /// A locally reduced batch proves its own reduction through its partial
    /// output. A hash-partitioned batch that did not flush proves low key
    /// cardinality through partition occupancy instead: keys hash uniformly,
    /// so a batch that touched only a small fraction of the partitions held
    /// few distinct keys.
    fn local_reduction_effective(&self, policy: BoundedSourceBatchingPolicy, device_budget_bytes: u64) -> Option<bool> {
        let sample = self.last?;
        if let Some(partial_rows) = sample.partial_rows {
            return Some(policy.source_adjacent_local_reduction_effective(
                sample.rows,
                partial_rows,
                sample.partial_bytes,
                device_budget_bytes,
            ));
        }
        Some(match (sample.non_empty_partitions, sample.partition_count) {
            (Some(non_empty), Some(partition_count)) => {
                policy.source_adjacent_local_reduction_from_partition_occupancy(non_empty, partition_count)
            },
            _ => false,
        })
    }
}

/// The partitioned path's coordinator loop.
///
/// Each iteration sizes one raw batch from the feedback and the attempt's
/// current allocation (`attempt_allocation_domain_usage`, which counts every
/// live allocation of the attempt including retained partition slices),
/// reads and partitions it, flushes the partitions the batching policy marks
/// ready, then records the batch's samples. The first batch runs whole and
/// fixes the partition count from its first raw input
/// (`accumulation_partition_count`); a later batch wider than one chunk per
/// lane is split across lanes, which all use that count. Retained
/// partition inputs stay on the coordinator at all times. Source commits are
/// counted as outputs are accepted and recorded once before the final merge.
fn execute_partitioned_source_adjacent_aggregate(
    execution: &mut BoundedExecutionAttempt<'_>,
    bounded: &BoundedPlan<'_>,
    chunks: &[SourceChunk],
    dispatch: Option<&AggregateLaneDispatch>,
) -> Result<CompiledGpuResource> {
    let owned = dispatch.map(|_| Arc::new(OwnedSourceAdjacentAggregate::new(bounded)));
    let partitioner = source_adjacent_aggregate_partitioner(PartitionedAggregate::of_plan(bounded));
    let estimated_input_bytes = chunks.iter().filter_map(SourceChunk::planning_bytes).sum::<u64>().max(1);
    let batching_policy = BoundedSourceBatchingPolicy::default();
    let device_budget_bytes = execution.device_budget_bytes();
    let mut partition_count = None;
    let mut partition_inputs: Option<Vec<Vec<SourceAdjacentPartitionInput>>> = None;
    let mut hold = execution.retained_state_charge();
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
        let retained_device_bytes =
            execution.context().attempt_allocation_domain_usage()?.map_or(0, |usage| usage.current_bytes);
        let row_cap =
            feedback.next_batch_row_cap(batching_policy, device_budget_bytes, retained_device_bytes, remaining_chunks);
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
        // The first raw input fixes the partition count from its own rows, so the batch that holds
        // it runs whole: lanes that each staged a first input could fix different counts.
        let lane_width = if partition_count.is_some() { dispatch.map_or(1, AggregateLaneDispatch::width) } else { 1 };
        let lane_batches = split_partitioned_raw_batch_for_lanes(batch, lane_width);
        if lane_batches.len() > 1 {
            let dispatch = dispatch.expect("multiple raw batches require aggregate lanes");
            let worker_owned = Arc::clone(owned.as_ref().expect("lane dispatch owns its source descriptor"));
            let worker_partitioner = partitioner.clone();
            let min_partition_count = source_adjacent_partitioned_aggregate_min_partition_count(chunks.len());
            let mut batches = lane_batches.into_iter();
            let mut samples = Vec::new();
            let result = dispatch.execute_work(
                execution,
                &mut batches,
                |batches, _| Ok(batches.next()),
                move |execution, batch| {
                    process_partitioned_source_adjacent_batch(
                        execution,
                        &worker_owned.plan(),
                        PartitionedSourceBatchWork {
                            batch,
                            pending: None,
                            hold: None,
                            partition_count,
                            local_reduction,
                            estimated_input_bytes,
                            min_partition_count,
                        },
                        &worker_partitioner,
                    )
                },
                |_, execution, output| {
                    if let Some(count) = output.partition_count {
                        debug_assert!(partition_count.is_none_or(|previous| previous == count));
                        partition_count = Some(count);
                    }
                    if let Some(pending) = output.pending {
                        let inputs =
                            partition_inputs.get_or_insert_with(|| (0..pending.len()).map(|_| Vec::new()).collect());
                        if inputs.len() != pending.len() {
                            return Err(bounded_invariant_violation(
                                "source-adjacent aggregate lanes returned inconsistent partition counts",
                                "source_adjacent_lane_partition_count_mismatch",
                            )
                            .with_fact("expected_partition_count", inputs.len())
                            .with_fact("actual_partition_count", pending.len()));
                        }
                        for (inputs, mut new_inputs) in inputs.iter_mut().zip(pending) {
                            inputs.append(&mut new_inputs);
                        }
                    }
                    final_merge.accept(execution, output.outputs)?;
                    let (chunk_count, row_group_count) = output.commit;
                    committed_source_chunks = committed_source_chunks.saturating_add(chunk_count);
                    committed_row_groups = committed_row_groups.saturating_add(row_group_count);
                    samples.extend(output.samples);
                    Ok(())
                },
            );
            result?;
            if let Some(inputs) = &mut partition_inputs {
                let flush_all = !source_adjacent_hold_pending_partitions(execution.context(), inputs, &mut hold);
                let ready = inputs
                    .iter_mut()
                    .map(|inputs| {
                        if flush_all || source_adjacent_partition_batch_ready(execution.context(), inputs) {
                            std::mem::take(inputs)
                        } else {
                            Vec::new()
                        }
                    })
                    .collect::<Vec<_>>();
                if ready.iter().any(|inputs| !inputs.is_empty()) {
                    let outputs = execute_partitioned_source_adjacent_final_work(
                        execution,
                        bounded,
                        ready,
                        partition_count.unwrap_or(1),
                        &partitioner,
                        Some(dispatch),
                        owned.as_ref(),
                    )?;
                    source_adjacent_hold_pending_partitions(execution.context(), inputs, &mut hold);
                    final_merge.accept(execution, outputs)?;
                }
            }
            for sample in samples {
                feedback.observe(sample);
            }
            execution.metrics().close_budget_window(&batch_task_id, budget_window);
            continue;
        }
        let work = PartitionedSourceBatchWork {
            batch: lane_batches.into_iter().next().expect("one unsplit raw batch remains"),
            pending: partition_inputs.take(),
            hold: hold.take(),
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
        hold = output.hold;
        final_merge.accept(execution, output.outputs)?;
        let (chunk_count, row_group_count) = output.commit;
        committed_source_chunks = committed_source_chunks.saturating_add(chunk_count);
        committed_row_groups = committed_row_groups.saturating_add(row_group_count);
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
    // The final work consumed the held slices, and the merge below holds its
    // own slices on the same ledger.
    drop(hold);
    match final_outputs {
        Ok(outputs) => final_merge.accept(execution, outputs)?,
        Err(error) => {
            return Err(error);
        },
    }
    execution.metrics().record_source_chunk_commit_count(committed_source_chunks, committed_row_groups);
    final_merge.finish(execution, bounded)
}

/// Whether a grouped aggregate over a resident frame stream goes through the partitioned
/// accumulation (`execute_partitioned_frame_stream_aggregate`).
///
/// The aggregate must have a shape the partitioned accumulation accepts and keys that partition by
/// group key. The stream must be lazy, whose frame count is unknown, or hold more than one frame;
/// an eager single frame keeps the single-frame path.
pub(in crate::exec::bounded) fn should_execute_partitioned_frame_stream_aggregate(
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &ArrowSchemaRef,
    output_schema: &ArrowSchemaRef,
    inputs: &ResidentFrameStreamPayload,
) -> bool {
    partitioned_accumulation_accepts(&aggregate.spec)
        && (inputs.is_lazy() || inputs.len() > 1)
        && source_adjacent_aggregate_partitioner(PartitionedAggregate {
            node_id,
            aggregate,
            input_schema,
            output_schema,
        })
        .group_key_columns()
        .is_some()
}

/// Aggregates a resident frame stream through the partitioned accumulation.
///
/// Frames are pulled one at a time and each is staged like a raw source input. The key sample of
/// the first frame, then the feedback of the frames before, decides whether a frame is reduced
/// whole and its partial partitioned, or hash-partitioned so that its slices join the held
/// partition inputs. A frame is consumed by the step that takes it, so the consumer holds the
/// current frame, its partition copy and the held slices, and never pulls ahead. The first frame
/// fixes the partition count (`first_batch_partition_count`): an eager stream reports every frame
/// up front, and a lazy stream scales the first frame to its declared rows. The released partials
/// reach the aligned incremental merge, which finalizes them like the stream's other merges.
pub(in crate::exec::bounded) fn execute_partitioned_frame_stream_aggregate(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &ArrowSchemaRef,
    output_schema: &ArrowSchemaRef,
    inputs: &mut ResidentFrameStreamPayload,
    preserve_output_stream: bool,
) -> Result<CompiledGpuResource> {
    let target = PartitionedAggregate { node_id, aggregate, input_schema, output_schema };
    let partitioner = source_adjacent_aggregate_partitioner(target);
    let Some(key_columns) = partitioner.group_key_columns() else {
        return Err(bounded_invariant_violation(
            "a frame stream reached the partitioned aggregate without group-key partitions",
            "frame_stream_partitioned_aggregate_without_group_keys",
        )
        .with_fact("node_id", node_id));
    };
    let mut merge = CompiledIncrementalPartitionedMerge::new_aligned(
        execution.context(),
        node_id,
        aggregate,
        input_schema,
        output_schema,
        key_columns,
    )?;
    let batching_policy = BoundedSourceBatchingPolicy::default();
    let device_budget_bytes = execution.device_budget_bytes();
    let complete = !inputs.is_lazy();
    let total_bytes = inputs.total_bytes_hint();
    let expected_input_rows = Some(inputs.num_rows()).filter(|rows| *rows > 0);
    let expected_frames = inputs.len();
    let mut feedback = SourceAdjacentRawBatchFeedback::default();
    let mut partition_count = None;
    let mut pending = None;
    let mut hold = execution.retained_state_charge();
    let mut index = 0usize;
    while let Some(frame) = inputs.next_frame(execution) {
        let frame = frame?;
        execution.context().check_cancellation("query_engine.exec.bounded.frame_stream_partitioned_aggregate")?;
        let rows = frame.frame.num_rows() as u64;
        let bytes = frame.device_bytes();
        let current_partition_count = match partition_count {
            Some(count) => count,
            None => {
                let input_bytes_estimate = first_batch_estimated_total_bytes(
                    if complete { total_bytes.unwrap_or(bytes) } else { bytes },
                    1,
                    rows,
                    expected_input_rows,
                    expected_frames,
                    complete,
                );
                let count = accumulation_partition_count(
                    execution,
                    target,
                    &frame,
                    input_bytes_estimate,
                    source_adjacent_partitioned_aggregate_min_partition_count(expected_frames),
                )?;
                *partition_count.insert(count)
            },
        };
        let mut staging = SourceAdjacentRawStaging {
            local_reduction: feedback.local_reduction_effective(batching_policy, device_budget_bytes),
            partition_count: AccumulationPartitionCount::Fixed(current_partition_count),
            partitioner: &partitioner,
        };
        let raw =
            PartitionedRawInput { origin: PartitionedInputOrigin::Frame { index }, frame, rows, decoded_bytes: bytes };
        let piece = stage_source_adjacent_raw_input(execution, target, raw, &mut staging)?;
        let (outputs, sample) = accumulate_staged_raw_input(
            execution,
            target,
            &partitioner,
            current_partition_count,
            piece,
            &mut pending,
            &mut hold,
        )?;
        push_aligned_partials(execution, &mut merge, outputs)?;
        feedback.observe(sample);
        index += 1;
    }
    let outputs = process_partitioned_source_adjacent_final_batches(
        execution,
        target,
        pending.take().unwrap_or_default(),
        partition_count.unwrap_or(1),
        &partitioner,
    )?;
    // The final work consumed the held slices, and the merge below holds its
    // own slices on the same ledger.
    drop(hold);
    push_aligned_partials(execution, &mut merge, outputs)?;
    if preserve_output_stream {
        merge.finish_stream(execution).map(CompiledGpuResource::frame_stream_payload)
    } else {
        merge.finish(execution).map(CompiledGpuResource::new)
    }
}

/// Pushes partials that stay aligned to their group-key partitions into the aligned merge.
fn push_aligned_partials(
    execution: &mut BoundedExecutionAttempt<'_>,
    merge: &mut CompiledIncrementalPartitionedMerge<'_>,
    outputs: Vec<SourceAdjacentAlignedPartial>,
) -> Result<()> {
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
}

/// Splits one planned raw batch into up to `width` sub-batches by chunk so
/// every lane reads a share. A batch with fewer chunks than lanes stays whole;
/// the sub-batches keep consecutive batch indexes for the source read rows.
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

/// Reads one raw batch and turns each of its raw inputs into partition work.
///
/// A raw input is either reduced whole first (`local_reduction`, when the
/// previous batch or, for the first batch, the input's own key sample showed
/// few distinct keys) or hash-partitioned into pending slices; pending
/// partitions that the batching policy marks ready are aggregated here. The raw input's staged frame is consumed by whichever
/// step takes it, so at most one partitioned copy of a raw input is live.
/// Flushing ready partitions acts on slices retained across batches, so it
/// stays outside the raw batch's retry scope.
fn process_partitioned_source_adjacent_batch(
    execution: &mut BoundedExecutionAttempt<'_>,
    bounded: &BoundedPlan<'_>,
    work: PartitionedSourceBatchWork,
    partitioner: &SourceAdjacentAggregatePartitioner,
) -> Result<PartitionedSourceBatchOutput> {
    let PartitionedSourceBatchWork {
        batch,
        mut pending,
        mut hold,
        mut partition_count,
        local_reduction,
        estimated_input_bytes,
        min_partition_count,
    } = work;
    let planned_chunk_count = batch.chunk_count() as u64;
    let mut staging = SourceAdjacentRawStaging {
        local_reduction,
        partition_count: match partition_count {
            Some(count) => AccumulationPartitionCount::Fixed(count),
            None => AccumulationPartitionCount::FromFirstInput {
                input_bytes_estimate: estimated_input_bytes,
                min_partition_count,
            },
        },
        partitioner,
    };
    let pieces = execute_source_adjacent_raw_source_batch(execution, bounded, batch, 1, &mut staging)?;
    let AccumulationPartitionCount::Fixed(current_partition_count) = staging.partition_count else {
        return Err(bounded_invariant_violation(
            "a staged raw batch left the accumulation's partition count unset",
            "source_adjacent_partition_count_unset",
        ));
    };
    partition_count = Some(current_partition_count);
    let mut outputs = Vec::new();
    let mut samples = Vec::with_capacity(pieces.len());
    let mut committed_row_groups = 0u64;
    for piece in pieces {
        committed_row_groups = committed_row_groups.saturating_add(piece.origin.source_row_group_count());
        let (piece_outputs, sample) = accumulate_staged_raw_input(
            execution,
            PartitionedAggregate::of_plan(bounded),
            partitioner,
            current_partition_count,
            piece,
            &mut pending,
            &mut hold,
        )?;
        outputs.extend(piece_outputs);
        samples.push(sample);
    }
    // A batch split by retry still commits its planned chunks once, with the
    // row groups of every piece.
    let commit = (planned_chunk_count, committed_row_groups);
    Ok(PartitionedSourceBatchOutput { pending, hold, partition_count, outputs, samples, commit })
}

/// Adds one staged raw input to the accumulation and returns the partials it releases, with the
/// input's feedback sample.
///
/// A locally reduced input's partials are returned as they are. A partitioned input's slices join
/// the held partition inputs; then every partition is aggregated when the retained-state ledger
/// refuses to hold the slices past the partition batch byte limit, and otherwise the partitions
/// the readiness rules release. The hold is sized again to what stays. The sample also records
/// the released partials.
fn accumulate_staged_raw_input(
    execution: &mut BoundedExecutionAttempt<'_>,
    target: PartitionedAggregate<'_>,
    partitioner: &SourceAdjacentAggregatePartitioner,
    partition_count: usize,
    piece: SourceAdjacentRawPiece,
    pending: &mut Option<Vec<Vec<SourceAdjacentPartitionInput>>>,
    hold: &mut Option<RetentionCharge>,
) -> Result<(Vec<SourceAdjacentAlignedPartial>, SourceAdjacentRawBatchSample)> {
    let SourceAdjacentRawPiece { origin, mut sample, staged } = piece;
    let partitions = match staged {
        SourceAdjacentRawStaged::Reduced(reduced) => {
            observe_aligned_partial_outputs(&mut sample, &reduced);
            return Ok((reduced, sample));
        },
        SourceAdjacentRawStaged::Partitioned(partitions) => partitions,
    };
    let inputs = pending.get_or_insert_with(|| (0..partition_count).map(|_| Vec::new()).collect());
    for (partition_index, partition) in partitions.into_iter().enumerate() {
        if let Some(partition) = partition {
            // Oversized partitions still enter the budgeted partial retry.
            inputs[partition_index].push(SourceAdjacentPartitionInput { origin: origin.clone(), partial: partition });
        }
    }
    let mut outputs = Vec::new();
    let flush_all = !source_adjacent_hold_pending_partitions(execution.context(), inputs, hold);
    for (partition_index, partition_inputs) in inputs.iter_mut().enumerate() {
        if partition_inputs.is_empty()
            || (!flush_all && !source_adjacent_partition_batch_ready(execution.context(), partition_inputs))
        {
            continue;
        }
        let reduced = execute_source_adjacent_partition_aggregate_batch(
            execution,
            target,
            partition_index,
            partition_count,
            partitioner,
            partition_inputs,
        )?;
        observe_aligned_partial_outputs(&mut sample, &reduced);
        outputs.extend(reduced);
    }
    source_adjacent_hold_pending_partitions(execution.context(), inputs, hold);
    Ok((outputs, sample))
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
            PartitionedAggregate::of_plan(bounded),
            pending,
            partition_count,
            partitioner,
        );
    };
    let owned = Arc::clone(owned.expect("lane dispatch owns its source descriptor"));
    let partitioner = partitioner.clone();
    let batches = pending.into_iter().enumerate().filter(|(_, inputs)| !inputs.is_empty());
    let mut state = (batches, Vec::new());
    dispatch.execute_work(
        execution,
        &mut state,
        |state, _| Ok(state.0.next()),
        move |execution, (partition_index, mut inputs)| {
            execute_source_adjacent_partition_aggregate_batch(
                execution,
                PartitionedAggregate::of_plan(&owned.plan()),
                partition_index,
                partition_count,
                &partitioner,
                &mut inputs,
            )
        },
        |state, _, mut output| {
            state.1.append(&mut output);
            Ok(())
        },
    )?;
    Ok(state.1)
}

fn process_partitioned_source_adjacent_final_batches(
    execution: &mut BoundedExecutionAttempt<'_>,
    target: PartitionedAggregate<'_>,
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
            target,
            partition_index,
            partition_count,
            partitioner,
            &mut inputs,
        ) {
            Ok(mut partials) => outputs.append(&mut partials),
            Err(error) => {
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
    target: PartitionedAggregate<'_>,
    origin: &PartitionedInputOrigin,
    frame: CommittedPartial,
    partition_count: usize,
    partitioner: &SourceAdjacentAggregatePartitioner,
) -> Result<Vec<SourceAdjacentAlignedPartial>> {
    let input_evidence = PartialAggregateInputEvidence {
        source_adjacent_partition_strategy: "local_reduction",
        source_adjacent_max_partition_fan_in: Some(1),
        ..source_adjacent_partitioned_aggregate_input_evidence(
            std::slice::from_ref(origin),
            partitioner,
            partition_count,
            1,
            1,
        )
    };
    let partials = execute_source_adjacent_partition_aggregate_with_retry(
        execution,
        target,
        SOURCE_ADJACENT_LOCAL_REDUCTION_STAGE,
        0,
        frame,
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
    let output_key_columns = lower_final_aggregate_spec(&target.aggregate.spec)?.key_columns;
    let mut outputs = Vec::new();
    for (partial_index, partial) in partials.into_iter().enumerate() {
        let partitions = match partition_final_merge_partial(
            execution,
            target.node_id,
            partial_index,
            partial,
            &output_key_columns,
            partition_count,
        ) {
            Ok(partitions) => partitions,
            Err(error) => {
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

/// How a raw input is turned into partition work inside its retry scope.
/// `local_reduction` is `None` when no batch has been observed yet; the input's
/// key sample then decides.
struct SourceAdjacentRawStaging<'a> {
    local_reduction: Option<bool>,
    partition_count: AccumulationPartitionCount,
    partitioner: &'a SourceAdjacentAggregatePartitioner,
}

/// The accumulation's partition count as a raw input is staged.
#[derive(Debug, Clone, Copy)]
enum AccumulationPartitionCount {
    /// Fixed by an earlier raw input of the run.
    Fixed(usize),
    /// Fixed by the first raw input staged (`accumulation_partition_count`), for an input
    /// estimated at `input_bytes_estimate` bytes in all.
    FromFirstInput { input_bytes_estimate: u64, min_partition_count: usize },
}

/// Partition count of the partitioned accumulation, fixed from its first raw input.
///
/// A partition holds its input slices and produces the partial state of the same rows, so its
/// working set is bounded by the larger of the two. `input_bytes_estimate` estimates the whole
/// input. When the partial-state upper bound of the first input's rows
/// (`partial_aggregate_work_reservation_upper_bound_bytes`) exceeds that input's own bound under
/// the same byte model (`aggregate_input_upper_bound_bytes`), the estimate grows by that ratio.
/// The count is the aggregate partition count of the result, at least `min_partition_count`. A
/// group-only aggregate's state columns are a subset of its input columns, so its estimate stays
/// the input estimate. A first input without rows bounds nothing and leaves the estimate as it is.
fn accumulation_partition_count(
    execution: &mut BoundedExecutionAttempt<'_>,
    target: PartitionedAggregate<'_>,
    first_input: &CommittedPartial,
    input_bytes_estimate: u64,
    min_partition_count: usize,
) -> Result<usize> {
    let mut estimate = input_bytes_estimate;
    if first_input.frame.num_rows() > 0 {
        let state = partial_aggregate_work_reservation_upper_bound_bytes(
            execution,
            target.aggregate,
            target.input_schema,
            target.output_schema,
            &first_input.frame,
        )?;
        let input = aggregate_input_upper_bound_bytes(execution, target.input_schema, &first_input.frame)?;
        if let (Some(state), Some(input)) = (state, input)
            && state > input
        {
            estimate = (u128::from(input_bytes_estimate) * u128::from(state) / u128::from(input))
                .min(u128::from(u64::MAX)) as u64;
        }
    }
    Ok(partitioned_aggregate_partition_count(execution.context(), estimate).max(min_partition_count))
}

/// One raw input that completed its retry scope.
struct SourceAdjacentRawPiece {
    origin: PartitionedInputOrigin,
    sample: SourceAdjacentRawBatchSample,
    staged: SourceAdjacentRawStaged,
}

enum SourceAdjacentRawStaged {
    /// Partials of a locally reduced input, already aligned to the partitions.
    Reduced(Vec<SourceAdjacentAlignedPartial>),
    /// One slice per partition, `None` for an empty partition that is not kept.
    Partitioned(Vec<Option<CommittedPartial>>),
}

/// Reads a raw batch, runs the source-adjacent stages and partitions or
/// locally reduces the result, all in one retry scope.
///
/// A retryable resource failure anywhere in that scope drops the scope's
/// frames and splits the input: a multi-chunk batch by chunk, with `attempt`
/// reset to 1, and a singleton batch by row group while `attempt` is below
/// `max_retry_attempts` and the chunk holds more than
/// `min_retry_chunk_row_groups` row groups. Pieces return in file order and
/// stay private to the caller, which commits the planned batch once. Partial
/// aggregate metrics count executed work, so a local reduction that ran in a
/// failed attempt stays counted next to its retried pieces.
fn execute_source_adjacent_raw_source_batch(
    execution: &mut BoundedExecutionAttempt<'_>,
    bounded: &BoundedPlan<'_>,
    batch: SourceChunkBatch,
    attempt: usize,
    staging: &mut SourceAdjacentRawStaging<'_>,
) -> Result<Vec<SourceAdjacentRawPiece>> {
    execution.context().check_cancellation("query_engine.exec.bounded.source_adjacent_raw_source_batch_retry")?;
    let RetryLimits { max_attempts, min_split_row_groups } = RetryLimits::of(execution.context());
    let raw = if batch.is_singleton() {
        execute_bounded_raw_source_chunk(execution, bounded, &batch.chunks[0], attempt, max_attempts)
    } else {
        execute_bounded_raw_source_chunk_batch(execution, bounded, &batch)
    };
    let error = match raw.and_then(|raw| {
        let raw = PartitionedRawInput {
            origin: PartitionedInputOrigin::Source {
                chunks: batch.chunks.clone().into(),
                row_group_count: raw.source_row_group_count,
            },
            rows: raw.source_rows,
            decoded_bytes: raw.source_bytes,
            frame: raw.partial,
        };
        stage_source_adjacent_raw_input(execution, PartitionedAggregate::of_plan(bounded), raw, staging)
    }) {
        Ok(piece) => return Ok(vec![piece]),
        Err(error) if is_retryable_chunk_error(&error) => error,
        Err(error) => return Err(error),
    };
    record_chunk_retry_error(&error, execution.metrics());
    let children = if batch.is_singleton() {
        let chunk = &batch.chunks[0];
        let split = (attempt < max_attempts).then(|| split_retryable_chunk(chunk, min_split_row_groups)).flatten();
        let Some((left, right)) = split else {
            return Err(chunk_retry_facts(chunk, execution).attach_to_error(error));
        };
        let child = |chunk| SourceChunkBatch {
            batch_index: batch.batch_index,
            chunks: vec![chunk],
            estimated_source_read_bytes: None,
            source_batching_reason: "retry_split".to_string(),
        };
        (child(left), child(right), attempt + 1)
    } else {
        let Some((left, right)) = batch.split_for_retry() else {
            return Err(error);
        };
        (left, right, 1)
    };
    let (left, right, next_attempt) = children;
    execution.metrics().record_source_chunk_retry();
    backoff_before_retry(
        execution.context(),
        "query_engine.exec.bounded.source_adjacent_raw_source_batch_split_retry",
    )?;
    let mut pieces = execute_source_adjacent_raw_source_batch(execution, bounded, left, next_attempt, staging)?;
    pieces.append(&mut execute_source_adjacent_raw_source_batch(execution, bounded, right, next_attempt, staging)?);
    Ok(pieces)
}

/// Partitions or locally reduces one raw input, a staged source batch or a resident frame. A
/// source input's outputs are returned to its retry scope, never pushed into retained partition
/// state.
fn stage_source_adjacent_raw_input(
    execution: &mut BoundedExecutionAttempt<'_>,
    target: PartitionedAggregate<'_>,
    raw: PartitionedRawInput,
    staging: &mut SourceAdjacentRawStaging<'_>,
) -> Result<SourceAdjacentRawPiece> {
    let partition_count = match staging.partition_count {
        AccumulationPartitionCount::Fixed(count) => count,
        AccumulationPartitionCount::FromFirstInput { input_bytes_estimate, min_partition_count } => {
            let count =
                accumulation_partition_count(execution, target, &raw.frame, input_bytes_estimate, min_partition_count)?;
            staging.partition_count = AccumulationPartitionCount::Fixed(count);
            count
        },
    };
    let SourceAdjacentRawStaging { local_reduction, partitioner, .. } = *staging;
    let PartitionedRawInput { origin, frame, rows, decoded_bytes } = raw;
    let mut sample = SourceAdjacentRawBatchSample {
        rows,
        decoded_bytes,
        staged_bytes: frame.device_bytes(),
        partial_rows: None,
        partial_bytes: 0,
        non_empty_partitions: None,
        partition_count: None,
    };
    let local_reduction = match local_reduction {
        Some(local_reduction) => local_reduction,
        None => source_adjacent_key_sample_local_reduction(execution, target, &origin, &frame)?,
    };
    let staged = if local_reduction {
        SourceAdjacentRawStaged::Reduced(execute_source_adjacent_batch_local_reduction(
            execution,
            target,
            &origin,
            frame,
            partition_count,
            partitioner,
        )?)
    } else {
        let partitions = partition_source_adjacent_raw_chunk(
            execution,
            target.node_id,
            origin.chunk_index(),
            frame,
            partition_count,
            partitioner,
        )?;
        sample.non_empty_partitions = Some(
            partitions
                .iter()
                .filter(|partition| partition.as_ref().is_some_and(|partition| partition.frame.num_rows() > 0))
                .count(),
        );
        sample.partition_count = Some(partition_count);
        SourceAdjacentRawStaged::Partitioned(partitions)
    };
    Ok(SourceAdjacentRawPiece { origin, sample, staged })
}

const SOURCE_ADJACENT_KEY_SAMPLE_STRATEGY: &str = "source_adjacent_local_reduction_key_sample";

/// Decides local reduction for a raw input that no earlier batch has judged,
/// from the distinct group keys of a bounded row sample.
///
/// The sample gathers the group-key columns of every `stride`-th row, with the
/// stride chosen so at most `source_adjacent_local_reduction_key_sample_rows`
/// rows are taken, from an offset below the stride seeded by the chunk
/// fingerprint. Over the offset every row is sampled with probability
/// `1 / stride`, so every group appears with at least that probability and the
/// expected sampled group count is at least `groups / stride`. The sample's
/// ratio of groups to rows is therefore, in expectation, at least the input's
/// (up to the one row by which `sample_rows` can exceed `rows / stride`), and
/// the ppm threshold applied to an observed partial errs toward partitioning
/// when applied to the sample. A stride rather than a contiguous slice crosses
/// every key run of an input clustered by key, such as a file sorted on a group
/// column. The partial byte bound is checked against the staged bytes scaled by
/// the sampled ratio.
///
/// Group keys that do not lower to input columns leave the input partitioned.
fn source_adjacent_key_sample_local_reduction(
    execution: &mut BoundedExecutionAttempt<'_>,
    target: PartitionedAggregate<'_>,
    origin: &PartitionedInputOrigin,
    staged: &CommittedPartial,
) -> Result<bool> {
    let policy = BoundedSourceBatchingPolicy::default();
    let Ok(lowered_keys) = lower_aggregate_spec_keys(&target.aggregate.spec) else {
        return Ok(false);
    };
    let rows = staged.frame.num_rows() as u64;
    if rows == 0 {
        return Ok(false);
    }
    let stride = rows.div_ceil(policy.source_adjacent_local_reduction_key_sample_rows());
    let offset = origin.key_sample_seed() % stride;
    let sample_rows = (rows - offset).div_ceil(stride);
    let key_refs = lowered_keys.key_columns.iter().map(String::as_str).collect::<Vec<_>>();
    let token = execution_budget_token_for_boundary(
        execution.context(),
        execution.device_budget_bytes(),
        "source_adjacent_local_reduction_key_sample",
        "query_engine.bounded.aggregate.source_adjacent_key_sample.budget_token",
        "failed to create source-adjacent key sample budget token",
    )?;
    let (observation, plan_execution_id, parent_span_id) = stage_context_from_execution(execution.context());
    let stage_guard = RuntimeOperatorStageGuard::start(
        observation,
        plan_execution_id,
        parent_span_id,
        RuntimeOperatorStageKind::ExactCardinalityProbe,
        RuntimeOperatorStageIdentity::new()
            .with_node_id(target.node_id.to_string())
            .with_chunk_index(origin.chunk_index())
            .with_strategy_label(SOURCE_ADJACENT_KEY_SAMPLE_STRATEGY)
            .with_input_rows(sample_rows),
    );
    let sample_groups = count_source_adjacent_key_sample_groups(
        execution.context(),
        &token,
        staged.frame.frame(),
        &key_refs,
        offset,
        stride,
        sample_rows,
    );
    let sample_groups = match sample_groups {
        Ok(groups) => {
            stage_guard.finish(Some(groups), None);
            groups
        },
        Err(error) => {
            let error =
                error.with_fact_if_absent("node_id", target.node_id).with_fact_if_absent("sample_rows", sample_rows);
            stage_guard.fail(&error);
            return Err(error);
        },
    };
    let estimated_partial_bytes =
        u64::try_from(u128::from(staged.device_bytes()) * u128::from(sample_groups) / u128::from(sample_rows))
            .unwrap_or(u64::MAX);
    Ok(policy.source_adjacent_local_reduction_effective(
        sample_rows,
        sample_groups,
        estimated_partial_bytes,
        execution.device_budget_bytes(),
    ))
}

/// Distinct key rows among `sample_rows` rows of `frame`'s `key_columns`,
/// taken every `stride` rows from `offset`. A stride of 1 counts the whole
/// frame without a gather.
fn count_source_adjacent_key_sample_groups(
    context: &ExecutionContext,
    token: &ExecutionBudgetToken,
    frame: &cudf::DataFrame,
    key_columns: &[&str],
    offset: u64,
    stride: u64,
    sample_rows: u64,
) -> Result<u64> {
    let keys = frame.select(key_columns).map_err(|error| {
        cudf_query_error(
            "query_engine.bounded.aggregate.source_adjacent_key_sample.select",
            "failed to project source-adjacent key sample columns",
            error,
        )
    })?;
    let sample = if stride == 1 {
        keys
    } else {
        let (Ok(offset), Ok(stride), Ok(sample_rows)) =
            (i32::try_from(offset), i32::try_from(stride), usize::try_from(sample_rows))
        else {
            return Err(bounded_invariant_violation(
                "source-adjacent key sample positions exceed the cuDF row index range",
                "source_adjacent_key_sample_row_index_overflow",
            )
            .with_fact("input_rows", frame.num_rows()));
        };
        let positions = context.runtime()?.sequence(offset, stride, sample_rows).map_err(|error| {
            cudf_query_error(
                "query_engine.bounded.aggregate.source_adjacent_key_sample.positions",
                "failed to create source-adjacent key sample positions",
                error,
            )
        })?;
        keys.on(token.exec()).gather(&positions).map_err(|error| {
            attach_budget_token_failure_facts(
                cudf_query_error(
                    "query_engine.bounded.aggregate.source_adjacent_key_sample.gather",
                    "failed to gather source-adjacent key sample rows",
                    error,
                ),
                token,
            )
        })?
    };
    let distinct = sample
        .on(token.exec())
        .drop_duplicates(
            DropDuplicatesOptions::builder().subset(key_columns).keep(DuplicateKeepOption::KeepAny).build(),
        )
        .map_err(|error| {
            attach_budget_token_failure_facts(
                cudf_query_error(
                    "query_engine.bounded.aggregate.source_adjacent_key_sample.distinct",
                    "failed to count distinct source-adjacent key sample rows",
                    error,
                ),
                token,
            )
        })?;
    Ok(distinct.num_rows() as u64)
}

/// Plans the next raw source batch from the remaining chunks.
///
/// Row caps come from working-set feedback; the source-byte target remains
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

/// Batches `chunks` under the source-byte target and the row cap. Partition
/// columns, a scan limit, and a reader predicate cuDF cannot evaluate over a
/// multi-source read each force singleton batches, because the batched reader
/// path cannot reproduce them per file.
fn source_adjacent_raw_source_batches(
    memory_policy: MemoryControlPolicy,
    bounded: &BoundedPlan<'_>,
    chunks: &[SourceChunk],
    max_batch_rows: Option<u64>,
) -> Vec<SourceChunkBatch> {
    if !bounded.source.partition_columns.is_empty()
        || bounded.source.scan_limit.is_some()
        || bounded.source.reader_predicate.as_ref().is_some_and(|predicate| {
            !bounded.source.reader_predicate_is_supported(predicate, rapids_analysis_context(PlanningPolicy::default()))
        })
    {
        return singleton_source_chunk_batches(chunks);
    }
    // The working-set row target accounts for decoded input and partition copies;
    // the source-byte target also scales with this attempt's immutable grant.
    let grouped = is_source_adjacent_grouped_aggregate(bounded);
    let max_batch_source_read_bytes =
        BoundedSourceBatchingPolicy::default().source_adjacent_raw_source_batch_target_bytes(memory_policy, grouped);
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

/// Evidence for one partitioned partial aggregate over the inputs of `origins`. Source inputs
/// report their chunks; resident frames report no source and are not source-adjacent partials.
fn source_adjacent_partitioned_aggregate_input_evidence(
    origins: &[PartitionedInputOrigin],
    partitioner: &SourceAdjacentAggregatePartitioner,
    partition_count: usize,
    non_empty_partition_count: usize,
    max_partition_fan_in: usize,
) -> PartialAggregateInputEvidence {
    let chunks = origins
        .iter()
        .filter_map(|origin| match origin {
            PartitionedInputOrigin::Source { chunks, .. } => Some(chunks.iter().cloned()),
            PartitionedInputOrigin::Frame { .. } => None,
        })
        .flatten()
        .collect::<Vec<_>>();
    PartialAggregateInputEvidence {
        source_chunk_fingerprint: partial_aggregate_source_chunk_fingerprint(&chunks),
        source_adjacent_partial: !chunks.is_empty(),
        source_chunk_row_group_count: origins.iter().map(PartitionedInputOrigin::source_row_group_count).sum(),
        source_chunk_reads_all_row_groups: !chunks.is_empty() && chunks.iter().all(|chunk| chunk.reads_all_row_groups),
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

/// Sizes `hold` to the slices all retained partitions keep, and returns
/// whether they may stay. Their resident bytes share one query grant even when
/// individual partitions are small. Up to the partition batch byte limit the
/// slices stay uncharged. Past it they stay only while the attempt's
/// retained-state ledger accepts their bytes; a refusal, or an attempt without
/// a ledger, returns false with the hold unchanged, and the caller aggregates
/// every partition. Called again after partitions were aggregated, it only
/// shrinks the hold, which always succeeds.
fn source_adjacent_hold_pending_partitions(
    context: &ExecutionContext,
    inputs: &[Vec<SourceAdjacentPartitionInput>],
    hold: &mut Option<RetentionCharge>,
) -> bool {
    let bytes = inputs.iter().flatten().fold(0u64, |bytes, input| bytes.saturating_add(input.partial.device_bytes()));
    let limit = BoundedSourceBatchingPolicy::default()
        .source_adjacent_partition_batch_byte_limit_for_budget(context.device_budget_bytes());
    let charged = if bytes >= limit { bytes } else { 0 };
    hold.as_mut().map_or(charged == 0, |hold| hold.try_resize(charged))
}

fn source_adjacent_partition_batch_ready(context: &ExecutionContext, inputs: &[SourceAdjacentPartitionInput]) -> bool {
    BoundedSourceBatchingPolicy::default().source_adjacent_partition_batch_ready(
        inputs.len(),
        source_adjacent_partition_batch_observed_device_bytes(inputs),
        context.device_budget_bytes(),
    )
}

/// Resident bytes of a partition's pending slices, reported as zero for a
/// single slice so the byte bound alone never flushes a partition with one
/// input; the row-group bound still can.
fn source_adjacent_partition_batch_observed_device_bytes(inputs: &[SourceAdjacentPartitionInput]) -> u64 {
    if inputs.len() <= 1 {
        return 0;
    }
    inputs
        .iter()
        .map(|input| input.partial.device_bytes())
        .try_fold(0u64, |total, bytes| total.checked_add(bytes))
        .unwrap_or(u64::MAX)
}

/// Aggregates one partition's pending slices into partials. The slices are
/// taken out of `inputs` first and concatenated into one frame, so a failure
/// leaves the partition empty. The caller treats the error as terminal for the
/// batch.
fn execute_source_adjacent_partition_aggregate_batch(
    execution: &mut BoundedExecutionAttempt<'_>,
    target: PartitionedAggregate<'_>,
    partition_index: usize,
    partition_count: usize,
    partitioner: &SourceAdjacentAggregatePartitioner,
    inputs: &mut Vec<SourceAdjacentPartitionInput>,
) -> Result<Vec<SourceAdjacentAlignedPartial>> {
    let batch_inputs = std::mem::take(inputs);
    let non_empty_partition_count = usize::from(batch_inputs.iter().any(|input| input.partial.frame.num_rows() > 0));
    let max_partition_fan_in = batch_inputs.len();
    let origins = batch_inputs.iter().map(|input| input.origin.clone()).collect::<Vec<_>>();
    let input_evidence = source_adjacent_partitioned_aggregate_input_evidence(
        &origins,
        partitioner,
        partition_count,
        non_empty_partition_count,
        max_partition_fan_in,
    );
    let partials = batch_inputs.into_iter().map(|input| input.partial).collect::<Vec<_>>();
    let partition_input = concatenate_partition_inputs(execution, target.node_id, partition_index, partials)?;
    let partials = execute_source_adjacent_partition_aggregate_with_retry(
        execution,
        target,
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

/// Group-key partitioning requires every key to be an integer type, or at
/// least one key to be a string with the rest integers; keys that cannot be
/// lowered or include other types fall back to row balancing.
fn source_adjacent_aggregate_partitioner(target: PartitionedAggregate<'_>) -> SourceAdjacentAggregatePartitioner {
    let Ok(lowered_keys) = lower_aggregate_spec_keys(&target.aggregate.spec) else {
        return SourceAdjacentAggregatePartitioner::RowBalanced;
    };
    if lowered_keys.key_columns.is_empty() {
        return SourceAdjacentAggregatePartitioner::RowBalanced;
    }
    let mut has_string_key = false;
    let mut integer_keys = true;
    for key_column in &lowered_keys.key_columns {
        let Ok(field) = target.input_schema.field_with_name(key_column) else {
            return SourceAdjacentAggregatePartitioner::RowBalanced;
        };
        has_string_key |=
            matches!(field.data_type(), ArrowDataType::Utf8 | ArrowDataType::LargeUtf8 | ArrowDataType::Utf8View);
        integer_keys &= matches!(
            field.data_type(),
            ArrowDataType::Int8
                | ArrowDataType::Int16
                | ArrowDataType::Int32
                | ArrowDataType::Int64
                | ArrowDataType::UInt8
                | ArrowDataType::UInt16
                | ArrowDataType::UInt32
                | ArrowDataType::UInt64
        );
    }
    if !has_string_key && !integer_keys {
        return SourceAdjacentAggregatePartitioner::RowBalanced;
    }
    // Integer group equality, including null keys, is preserved by the raw-key
    // hash. Floating keys keep the existing path until signed-zero and NaN
    // canonicalization is covered by the alignment contract.
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
                return Err(error);
            },
        }
    }
    Ok(partitions)
}

/// Hash-partitions one staged raw input under a budget token.
///
/// Row balancing appends a row-position column and hashes that, which spreads
/// rows uniformly regardless of key skew; `original_columns` records the
/// columns to keep so the slice step removes the helper. The returned token
/// stays live through slicing so the partition copies are charged to the same
/// boundary as the partition itself.
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
            let with_partition_column =
                raw_frame.on(token.exec()).with_position_column(&partition_column).map_err(|error| {
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
                .on(token.exec())
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
            .on(token.exec())
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
        bounded_invariant_violation(
            "source-adjacent aggregate partition is missing a partition start offset",
            "partition_offset_missing",
        )
    })?;
    let end = *offsets.get(partition + 1).ok_or_else(|| {
        bounded_invariant_violation(
            "source-adjacent aggregate partition is missing a partition end offset",
            "partition_offset_missing",
        )
    })?;
    let len = end.saturating_sub(start);
    if len == 0 && !retain_empty {
        return Ok(None);
    }
    let sliced = budgeted_value(
        token.run("dataframe::slice", |exec| frame.on(exec).slice(start, len)).map_err(|error| {
            cudf_query_error(
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
        "source_adjacent_aggregate_partition_concat",
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
    target: PartitionedAggregate<'_>,
    stage: &'static str,
    partition_index: usize,
    input: CommittedPartial,
    input_evidence: PartialAggregateInputEvidence,
    retry_budget: usize,
) -> Result<Vec<CommittedPartial>> {
    let upper_bound_bytes = match partial_aggregate_work_reservation_upper_bound_bytes(
        execution,
        target.aggregate,
        target.input_schema,
        target.output_schema,
        &input.frame,
    ) {
        Ok(bytes) => bytes,
        Err(error) => {
            return Err(source_adjacent_partition_aggregate_error(error, target.node_id, stage, partition_index));
        },
    };
    if source_adjacent_partition_aggregate_should_split(
        input.frame.num_rows(),
        upper_bound_bytes,
        retry_budget,
        execution.device_budget_bytes(),
    ) {
        // A proactive split has no allocation failure behind it: no retry
        // counter, no backoff, only the cancellation boundary.
        if let Err(error) = execution
            .context()
            .check_cancellation("query_engine.exec.bounded.source_adjacent_partition_aggregate_split")
        {
            return Err(source_adjacent_partition_aggregate_error(error, target.node_id, stage, partition_index));
        }
        let (left, right) =
            split_source_adjacent_partition_aggregate_input(execution, target.node_id, stage, partition_index, input)?;
        let mut partials = match execute_source_adjacent_partition_aggregate_with_retry(
            execution,
            target,
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
            target,
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
            Err(error) => Err(error),
        }
    } else {
        execute_source_adjacent_partition_aggregate(execution, target, stage, partition_index, input, input_evidence)
            .map(|partial| vec![partial])
    }
}

fn execute_source_adjacent_partition_aggregate(
    execution: &mut BoundedExecutionAttempt<'_>,
    target: PartitionedAggregate<'_>,
    stage: &'static str,
    partition_index: usize,
    input: CommittedPartial,
    input_evidence: PartialAggregateInputEvidence,
) -> Result<CommittedPartial> {
    let input_rows = input.frame.num_rows() as u64;
    let input_frame = input.frame;
    let task_id = source_adjacent_aggregate_task_id(target.node_id, stage, format!("partition{partition_index}"));
    let token = match execution_budget_token_for_boundary(
        execution.context(),
        execution.device_budget_bytes(),
        stage,
        "query_engine.bounded.aggregate.source_adjacent_partial.budget_token",
        "failed to create source-adjacent partial aggregate budget token",
    ) {
        Ok(token) => token,
        Err(error) => {
            return Err(source_adjacent_partition_aggregate_error(error, target.node_id, stage, partition_index));
        },
    };
    let chunk_operator_timer = Instant::now();
    let budget_window = execution.metrics().open_budget_window();
    let result = execute_partial_aggregate_with_budget(
        execution.context(),
        Some(target.node_id),
        target.aggregate,
        target.input_schema,
        target.output_schema,
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
            return Err(source_adjacent_partition_aggregate_error(error, target.node_id, stage, partition_index));
        },
    };
    let output_bytes = match observe_device_memory_bytes(&frame, execution.metrics()) {
        Ok(bytes) => bytes,
        Err(error) => {
            return Err(source_adjacent_partition_aggregate_error(error, target.node_id, stage, partition_index));
        },
    };
    let output = ResidentGpuFrame::with_device_bytes(frame, output_bytes);
    execution.metrics().record_partial_aggregate(input_rows, output.frame.num_rows() as u64, output.device_bytes());
    Ok(output)
}

/// Proactive splits allowed per partition: one fewer than the retry attempts,
/// so the depth of the split tree matches the depth a failure-driven retry
/// could reach.
fn source_adjacent_partition_aggregate_retry_budget(context: &ExecutionContext) -> usize {
    RetryLimits::of(context).max_attempts - 1
}

/// Splits a partition proactively when its partial state can outgrow the
/// budget. Without a bound the partition runs whole. Callers pass the
/// attempt's immutable cap as `remaining_bytes`, so the comparison is against
/// the whole cap; the bytes currently free under it are not consulted.
fn source_adjacent_partition_aggregate_should_split(
    input_rows: usize,
    upper_bound_bytes: Option<u64>,
    retry_budget: usize,
    remaining_bytes: u64,
) -> bool {
    retry_budget > 0 && input_rows > 1 && upper_bound_bytes.is_some_and(|bytes| bytes > remaining_bytes)
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
            bounded_invariant_violation(
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
    let exec_runtime = match execution.context().runtime() {
        Ok(runtime) => runtime,
        Err(error) => {
            return Err(source_adjacent_partition_aggregate_error(error, node_id, stage, partition_index));
        },
    };
    let overrides = exec_runtime.exec();
    let left = match source_frame.on(overrides).slice(0, split_at) {
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
    let right = match source_frame.on(overrides).slice(split_at, right_len) {
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
    stage: &'static str,
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
            return Err(error);
        },
    };

    let first = inputs.remove(0);
    let first_frame = first.frame.into_frame();
    let concatenated = {
        let other_frames = inputs.iter().map(|input| input.frame.frame().clone()).collect::<Vec<_>>();
        let other_refs = other_frames.iter().collect::<Vec<_>>();
        first_frame.on(token.exec()).concat_many(&other_refs).map_err(|error| {
            attach_budget_token_failure_facts(
                cudf_query_error(concat_operation, concat_message, error)
                    .with_fact("node_id", node_id)
                    .with_fact("partition", partition_index)
                    .with_fact("input_count", inputs.len() + 1),
                &token,
            )
        })
    };

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
    Ok(ResidentGpuFrame::with_device_bytes(concatenated, output_bytes))
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
    fn source_adjacent_partial_aggregate_split_requires_budget_pressure_and_a_safe_retry() {
        assert!(!source_adjacent_partition_aggregate_should_split(1, Some(101), 1, 100));
        assert!(!source_adjacent_partition_aggregate_should_split(2, Some(101), 0, 100));
        assert!(!source_adjacent_partition_aggregate_should_split(2, Some(100), 1, 100));
        assert!(!source_adjacent_partition_aggregate_should_split(2, None, 1, 100));
        assert!(source_adjacent_partition_aggregate_should_split(2, Some(101), 1, 100));
    }

    #[test]
    fn grouped_aggregate_raw_batches_respect_grouped_target() {
        use std::sync::Arc;

        use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
        use query_planning::{
            execution_contracts::SourceChunkPlanningMode,
            expr::{AggregateSpec, Expr, LiteralValue},
            plan::{AggregateNode, FilterNode, NodeId},
            source::{LocalParquetFile, LocalParquetSource, ParquetSourceLocation, ResolvedParquetSource},
        };

        const CHUNK_BYTES: u64 = 300 * 1024 * 1024;
        const GROUPED_TARGET_BYTES: u64 = 94 * 1024 * 1024 * 1024 / 8;

        fn batchable_chunk(index: usize, bytes: u64) -> SourceChunk {
            SourceChunk {
                source_node_id: NodeId::from_index_for_test(1),
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
                estimated_decoded_bytes: None,
                estimated_column_elements: Some(1),
                max_column_elements: u64::MAX,
                exact_source_read_preflight_bytes: Some(bytes),
                source_file: Arc::default(),
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
                source_id: NodeId::from_index_for_test(1),
                source,
                pre_filter_projections: Vec::new(),
                filter: Some(filter),
                post_filter_projections: Vec::new(),
                aggregate_id: NodeId::from_index_for_test(3),
                aggregate_input_schema: schema,
                aggregate: (aggregate, schema),
                post_aggregate_stages: Vec::new(),
                sort: None,
                limit: None,
            }
        }

        let schema: SchemaRef = Arc::new(Schema::new(vec![Field::new("k", DataType::Utf8, true)]));
        let source = LocalParquetSource::from_files(
            vec![LocalParquetFile::new("/tmp/grouped.parquet")],
            Arc::clone(&schema),
            vec!["k".to_string()],
        );
        let filter =
            FilterNode { input: NodeId::from_index_for_test(1), predicate: Expr::Literal(LiteralValue::Boolean(true)) };
        let grouped = AggregateNode {
            input: NodeId::from_index_for_test(2),
            spec: AggregateSpec::new(vec![Expr::Literal(LiteralValue::Utf8("k".to_string()))], Vec::new()),
        };
        let ungrouped =
            AggregateNode { input: NodeId::from_index_for_test(2), spec: AggregateSpec::new(Vec::new(), Vec::new()) };
        let policy = MemoryControlPolicy {
            device_budget_bytes: 94 * 1024 * 1024 * 1024,
            max_source_chunk_bytes: 32 * 1024 * 1024 * 1024,
            max_row_groups_per_chunk: 1024,
            max_retry_attempts: 4,
            min_retry_chunk_row_groups: 1,
        };
        let chunks: Vec<SourceChunk> = (0..8).map(|index| batchable_chunk(index, CHUNK_BYTES)).collect();

        let grouped_batches = source_adjacent_raw_source_batches(
            policy,
            &plan_with_group_exprs(&source, &filter, &grouped, &schema),
            &chunks,
            None,
        );
        assert_eq!(grouped_batches.len(), 1);
        for batch in &grouped_batches {
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
        assert_eq!(ungrouped_batches.len(), 1);

        let explicit = MemoryControlPolicy { max_source_chunk_bytes: CHUNK_BYTES, ..policy };
        let explicitly_limited = source_adjacent_raw_source_batches(
            explicit,
            &plan_with_group_exprs(&source, &filter, &grouped, &schema),
            &chunks,
            None,
        );
        assert_eq!(explicitly_limited.len(), chunks.len());

        let batching = BoundedSourceBatchingPolicy::default();
        let mut feedback = SourceAdjacentRawBatchFeedback::default();
        feedback.observe(SourceAdjacentRawBatchSample {
            rows: 1_000,
            decoded_bytes: 16_000,
            staged_bytes: 8_000,
            ..Default::default()
        });
        let narrow_rows = feedback.next_batch_row_cap(batching, 256 * 1024 * 1024, 0, &chunks).unwrap();
        feedback.observe(SourceAdjacentRawBatchSample {
            rows: 10,
            decoded_bytes: 10_240,
            staged_bytes: 10_240,
            ..Default::default()
        });
        let wide_rows = feedback.next_batch_row_cap(batching, 256 * 1024 * 1024, 0, &chunks).unwrap();
        assert!(
            wide_rows < narrow_rows / 32,
            "recent wide rows must reduce the next batch despite a long narrow history"
        );
        feedback.observe(SourceAdjacentRawBatchSample {
            rows: 1_000,
            decoded_bytes: 16_000,
            staged_bytes: 8_000,
            ..Default::default()
        });
        assert_eq!(feedback.next_batch_row_cap(batching, 256 * 1024 * 1024, 0, &chunks), Some(wide_rows));
        let retained_rows =
            feedback.next_batch_row_cap(batching, 256 * 1024 * 1024, 192 * 1024 * 1024, &chunks).unwrap();
        assert_eq!(retained_rows, wide_rows / 4);
        assert!(feedback.next_batch_row_cap(batching, 64 * 1024 * 1024 * 1024, 0, &chunks).unwrap() > wide_rows * 200);

        // A row cap from working-set feedback bounds a batch below the source
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
