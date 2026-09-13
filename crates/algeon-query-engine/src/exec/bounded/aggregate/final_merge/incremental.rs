use std::{
    collections::VecDeque,
    sync::Arc,
    time::{Duration, Instant},
};

use super::{
    AggregateExecutionStage, AggregateExecutionStrategy, AggregateNode, ArrowSchemaRef, BoundedExecutionAttempt,
    BoundedFinalMergePolicy, CommittedPartial, ExecutionContext, MemoryControlPolicy, NodeId, QueryError,
    QueryErrorCode, Result, aggregate_strategy_capability_decision,
    compiled::{
        execute_compiled_final_aggregate_projection, merge_compiled_partitioned_final_aggregate_partial_refs,
        record_compiled_partitioned_final_merge_strategy,
    },
    concatenate_partitioned_final_merge_outputs, execute_compiled_bounded_aggregate_from_empty_input,
    execution_budget_token_for_boundary, lower_final_aggregate_spec, observe_device_memory_bytes,
    partition_final_merge_partial, partitioned_aggregate_partition_count, reduce_merge_fan_in,
    release_partials_reverse,
};
use crate::{
    exec::{
        bounded::{
            BoundedPartitionTargetPolicy, ResidentGpuFrame,
            compiled::{ResidentFrameStreamPayload, ResidentFrameStreamSource},
        },
        retry::{backoff_before_retry, is_retryable_resource_error, record_retryable_resource_error},
    },
    runtime::resident::GpuDataFrame,
};

const MAX_PENDING_PARTIALS: usize = 16;

/// Whether a compiled aggregate over a partial stream should merge its partials
/// incrementally into group-key-aligned partition states instead of retaining
/// every partial until the stream ends.
pub(in crate::exec::bounded::aggregate) fn should_execute_incremental_partitioned_final_aggregate_merge(
    context: &ExecutionContext,
    aggregate: &AggregateNode,
    input_schema: &ArrowSchemaRef,
) -> bool {
    matches!(context.memory_control_policy(), MemoryControlPolicy::Bounded { .. })
        && aggregate.spec.grouping_sets.is_none()
        && aggregate_strategy_capability_decision(
            &aggregate.spec,
            input_schema.as_ref(),
            AggregateExecutionStage::FinalMerge,
        )
        .selected_strategy
            == AggregateExecutionStrategy::PartitionedMerge
}

/// Bounded incremental final merge for a compiled partial-aggregate stream.
///
/// Partials arrive one chunk at a time. They are batched, hash-partitioned by
/// the final group keys, and folded into one merged partial state per
/// partition once a partition's pending slices reach its merged state's size,
/// so the retained bytes never exceed the true merged state plus one batch of
/// pending slices and each merge copies at most twice the partition it folds.
/// Raw slices are first merged among themselves and the result is then merged
/// with the partition's earlier merged state, mirroring the levels of the
/// whole-set merge. Finalized partitions can be pulled independently;
/// consumers requiring a single frame concatenate them in partition order.
pub(in crate::exec::bounded::aggregate) struct CompiledIncrementalPartitionedMerge<'a> {
    node_id: NodeId,
    aggregate: &'a AggregateNode,
    input_schema: &'a ArrowSchemaRef,
    output_schema: &'a ArrowSchemaRef,
    key_columns: Vec<String>,
    aligned_key_columns: Option<Vec<String>>,
    expected_partials: usize,
    expected_input_rows: Option<u64>,
    partition_count: Option<usize>,
    partition_batch_bytes: u64,
    min_flush_bytes: u64,
    max_attempts: usize,
    pending_raw: Vec<CommittedPartial>,
    pending_raw_bytes: u64,
    pending: Vec<Vec<CommittedPartial>>,
    pending_bytes: Vec<u64>,
    /// Merged partial states per partition. Merged states carry the final
    /// merge's column shape, which differs from raw partial state for COUNT
    /// and AVG, so raw slices and merged states are never merged in one set.
    merged: Vec<Vec<CommittedPartial>>,
    input_rows: u64,
    partial_count: usize,
    flush_count: usize,
    max_partition_fan_in: usize,
    max_pending_partials: usize,
    merge_time: Duration,
}

pub(in crate::exec::bounded::aggregate) struct IncrementalPartitionStates {
    pub partitions: Vec<(usize, CommittedPartial)>,
    pub partition_count: usize,
    pub input_rows: u64,
    pub partial_count: usize,
    pub flush_count: usize,
    pub max_partition_fan_in: usize,
    pub max_pending_partials: usize,
    pub merge_time: Duration,
}

impl<'a> CompiledIncrementalPartitionedMerge<'a> {
    /// `expected_partials` is the stream's known chunk count (or eager frame
    /// count) and `expected_input_rows` its known row total when the producer
    /// declared one; both only seed the partition count from the first batch.
    pub(in crate::exec::bounded::aggregate) fn new(
        context: &ExecutionContext,
        node_id: NodeId,
        aggregate: &'a AggregateNode,
        input_schema: &'a ArrowSchemaRef,
        output_schema: &'a ArrowSchemaRef,
        expected_partials: usize,
        expected_input_rows: Option<u64>,
    ) -> Result<Self> {
        let MemoryControlPolicy::Bounded { device_budget_bytes, max_retry_attempts, .. } =
            context.memory_control_policy()
        else {
            unreachable!("incremental partitioned final merge is created only for bounded policies");
        };
        let key_columns = lower_final_aggregate_spec(&aggregate.spec)?.key_columns;
        let policy = BoundedFinalMergePolicy::default();
        let partition_target_bytes =
            BoundedPartitionTargetPolicy::default().aggregate_partition_target_bytes(device_budget_bytes);
        Ok(Self {
            node_id,
            aggregate,
            input_schema,
            output_schema,
            key_columns,
            aligned_key_columns: None,
            expected_partials: expected_partials.max(1),
            expected_input_rows: expected_input_rows.filter(|rows| *rows > 0),
            partition_count: None,
            partition_batch_bytes: policy.incremental_partition_batch_bytes(partition_target_bytes),
            min_flush_bytes: policy.incremental_min_flush_bytes(partition_target_bytes),
            max_attempts: max_retry_attempts.max(1),
            pending_raw: Vec::new(),
            pending_raw_bytes: 0,
            pending: Vec::new(),
            pending_bytes: Vec::new(),
            merged: Vec::new(),
            input_rows: 0,
            partial_count: 0,
            flush_count: 0,
            max_partition_fan_in: 0,
            max_pending_partials: 0,
            merge_time: Duration::ZERO,
        })
    }

    pub(in crate::exec::bounded::aggregate) fn new_aligned(
        context: &ExecutionContext,
        node_id: NodeId,
        aggregate: &'a AggregateNode,
        input_schema: &'a ArrowSchemaRef,
        output_schema: &'a ArrowSchemaRef,
        key_columns: &[String],
    ) -> Result<Self> {
        let mut merge = Self::new(context, node_id, aggregate, input_schema, output_schema, 1, None)?;
        merge.aligned_key_columns = Some(key_columns.to_vec());
        Ok(merge)
    }

    pub(in crate::exec::bounded::aggregate) fn push_aligned(
        &mut self,
        execution: &mut BoundedExecutionAttempt<'_>,
        partition_index: usize,
        partition_count: usize,
        key_columns: &[String],
        partial: CommittedPartial,
    ) -> Result<()> {
        if self.partition_count.is_none() {
            self.partition_count = Some(partition_count);
            self.pending = (0..partition_count).map(|_| Vec::new()).collect();
            self.pending_bytes = vec![0; partition_count];
            self.merged = (0..partition_count).map(|_| Vec::new()).collect();
        }
        let expected_count = self.partition_count.expect("aligned merge has a fixed partition count");
        let expected_keys = self.aligned_key_columns.as_ref().expect("aligned merge has fixed keys");
        if partition_count != expected_count || partition_index >= expected_count || key_columns != expected_keys {
            return Err(QueryError::new(
                QueryErrorCode::Internal,
                "query_engine.final_merge.aligned_partition_metadata",
                "aligned group-key final merge received incompatible partition metadata",
            )
            .with_fact("expected_partition_count", expected_count)
            .with_fact("actual_partition_count", partition_count)
            .with_fact("partition_index", partition_index)
            .with_fact("expected_keys", expected_keys.join(","))
            .with_fact("actual_keys", key_columns.join(",")));
        }
        if partial.frame.num_rows() == 0 {
            return Ok(());
        }
        let bytes = partial.device_bytes();
        if !self.pending[partition_index].is_empty()
            && (self.pending[partition_index].len() >= MAX_PENDING_PARTIALS
                || self.pending_bytes[partition_index].saturating_add(bytes) > self.partition_batch_bytes)
        {
            self.flush_partition(execution, partition_index)?;
        }
        self.input_rows = self.input_rows.saturating_add(partial.frame.num_rows() as u64);
        self.partial_count += 1;
        self.pending_bytes[partition_index] = self.pending_bytes[partition_index].saturating_add(bytes);
        self.pending[partition_index].push(partial);
        self.max_pending_partials = self.max_pending_partials.max(self.pending[partition_index].len());
        self.flush_ready_partitions(execution)
    }

    pub(in crate::exec::bounded::aggregate) fn finish_partition_states(
        mut self,
        execution: &mut BoundedExecutionAttempt<'_>,
    ) -> Result<IncrementalPartitionStates> {
        if !self.pending_raw.is_empty() {
            self.partition_pending_raw(execution)?;
        }
        let partition_count = self.partition_count.unwrap_or(1);
        let mut partitions = Vec::new();
        for partition_index in 0..partition_count {
            self.flush_partition(execution, partition_index)?;
            if let Some(state) = self.merged[partition_index].pop() {
                partitions.push((partition_index, state));
            }
        }
        Ok(IncrementalPartitionStates {
            partitions,
            partition_count,
            input_rows: self.input_rows,
            partial_count: self.partial_count,
            flush_count: self.flush_count,
            max_partition_fan_in: self.max_partition_fan_in,
            max_pending_partials: self.max_pending_partials,
            merge_time: self.merge_time,
        })
    }

    /// Folds one partial into the merge.
    ///
    /// Pending work that the new partial would push over the batch threshold
    /// is merged before the partial is retained, so a failed merge leaves this
    /// partial unrecorded and the chunk boundary's retry can re-produce it
    /// without duplicating its rows.
    pub(in crate::exec::bounded::aggregate) fn push(
        &mut self,
        execution: &mut BoundedExecutionAttempt<'_>,
        partial: CommittedPartial,
    ) -> Result<()> {
        let partial_bytes = partial.device_bytes();
        if !self.pending_raw.is_empty()
            && (self.pending_raw.len() >= MAX_PENDING_PARTIALS
                || self.pending_raw_bytes.saturating_add(partial_bytes) > self.partition_batch_bytes)
        {
            self.partition_pending_raw(execution)?;
            self.flush_ready_partitions(execution)?;
        }
        self.input_rows = self.input_rows.saturating_add(partial.frame.num_rows() as u64);
        self.partial_count += 1;
        self.pending_raw_bytes = self.pending_raw_bytes.saturating_add(partial_bytes);
        self.pending_raw.push(partial);
        self.max_pending_partials = self.max_pending_partials.max(self.pending_raw.len());
        Ok(())
    }

    /// Merges everything still pending and returns the finalized output.
    pub(in crate::exec::bounded::aggregate) fn finish(
        self,
        execution: &mut BoundedExecutionAttempt<'_>,
    ) -> Result<ResidentGpuFrame> {
        let node_id = self.node_id;
        let frames = self.finish_partitions(execution, false)?.into_remaining_frames_owned(execution)?;
        concatenate_partitioned_final_merge_outputs(execution, node_id, frames)
    }

    pub(in crate::exec::bounded::aggregate) fn finish_stream(
        self,
        execution: &mut BoundedExecutionAttempt<'_>,
    ) -> Result<ResidentFrameStreamPayload> {
        self.finish_partitions(execution, true)
    }

    fn finish_partitions(
        mut self,
        execution: &mut BoundedExecutionAttempt<'_>,
        streamed_output: bool,
    ) -> Result<ResidentFrameStreamPayload> {
        if self.partial_count == 0 {
            return execute_compiled_bounded_aggregate_from_empty_input(
                execution,
                self.node_id,
                self.aggregate,
                self.input_schema,
                self.output_schema,
            )
            .map(|frame| ResidentFrameStreamPayload::new(vec![frame]));
        }
        // A single partial needs no merge, exactly as the whole-set path
        // projects it directly without recording a merge attempt.
        if self.partial_count == 1 && self.partition_count.is_none() {
            let partial = self.pending_raw.pop().expect("single partial is pending");
            return execute_compiled_final_aggregate_projection(
                execution,
                self.node_id,
                self.aggregate,
                self.input_schema,
                self.output_schema,
                partial,
            )
            .map(|frame| ResidentFrameStreamPayload::new(vec![frame]));
        }
        if !self.pending_raw.is_empty() {
            self.partition_pending_raw(execution)?;
        }
        let partition_count = self.partition_count.unwrap_or(1);
        let timer = Instant::now();
        let mut partitions = VecDeque::with_capacity(partition_count);
        let mut non_empty_partition_count = 0usize;
        for partition_index in 0..partition_count {
            self.flush_partition(execution, partition_index)?;
            let Some(merged) = self.merged[partition_index].pop() else {
                continue;
            };
            debug_assert!(self.merged[partition_index].is_empty(), "flush leaves one merged state per partition");
            non_empty_partition_count += 1;
            partitions.push_back(merged);
        }
        if partitions.is_empty() {
            return execute_compiled_bounded_aggregate_from_empty_input(
                execution,
                self.node_id,
                self.aggregate,
                self.input_schema,
                self.output_schema,
            )
            .map(|frame| ResidentFrameStreamPayload::new(vec![frame]));
        }
        execution.metrics().record_final_merge_attempt(self.max_partition_fan_in.max(1));
        execution.metrics().record_final_merge_level();
        let elapsed = self.merge_time.saturating_add(timer.elapsed());
        let output_rows = partitions.iter().map(|partial| partial.frame.num_rows() as u64).sum();
        record_compiled_partitioned_final_merge_strategy(
            execution.context(),
            self.node_id,
            self.aggregate,
            self.input_schema,
            self.input_rows,
            output_rows,
            partition_count,
            non_empty_partition_count,
            self.max_partition_fan_in,
            elapsed,
            [
                ("partitioned_merge_incremental", "true".to_string()),
                ("partitioned_merge_incremental_partials", self.partial_count.to_string()),
                ("partitioned_merge_incremental_flushes", self.flush_count.to_string()),
                ("partitioned_merge_incremental_max_pending_partials", self.max_pending_partials.to_string()),
                ("partitioned_merge_streamed_finalization", streamed_output.to_string()),
            ],
            execution.metrics(),
        );
        let retained_bytes = partitions.iter().map(CommittedPartial::device_bytes).sum();
        Ok(ResidentFrameStreamPayload::from_producer(
            // The allocator cap bounds each eventual output; retained partial
            // bytes are observed separately and never used as an allocation veto.
            execution.device_budget_bytes(),
            retained_bytes,
            output_rows,
            self.output_schema.fields().len(),
            Box::new(PartitionedFinalAggregateStream {
                node_id: self.node_id,
                aggregate: self.aggregate.clone(),
                input_schema: Arc::clone(self.input_schema),
                output_schema: Arc::clone(self.output_schema),
                partitions,
            }),
        ))
    }

    /// Partition count from the first batch: its partial bytes scaled to the
    /// declared input rows when known, else to the expected partial count.
    fn partition_count_for_first_batch(&self, context: &ExecutionContext) -> usize {
        let estimated_total_bytes = match self.expected_input_rows {
            Some(rows) if self.input_rows > 0 => (u128::from(self.pending_raw_bytes) * u128::from(rows)
                / u128::from(self.input_rows))
            .min(u128::from(u64::MAX)) as u64,
            _ => self
                .pending_raw_bytes
                .div_ceil(self.pending_raw.len().max(1) as u64)
                .saturating_mul(self.expected_partials as u64),
        }
        .max(1);
        partitioned_aggregate_partition_count(context, estimated_total_bytes).max(
            BoundedFinalMergePolicy::default().partitioned_final_merge_partial_partition_floor(self.expected_partials),
        )
    }

    /// Concatenates the pending raw partials and hash-partitions them once, so
    /// each partition receives one aligned slice per batch instead of one per chunk.
    fn partition_pending_raw(&mut self, execution: &mut BoundedExecutionAttempt<'_>) -> Result<()> {
        let partition_count = match self.partition_count {
            Some(count) => count,
            None => {
                let count = self.partition_count_for_first_batch(execution.context());
                self.partition_count = Some(count);
                self.pending = (0..count).map(|_| Vec::new()).collect();
                self.pending_bytes = vec![0; count];
                self.merged = (0..count).map(|_| Vec::new()).collect();
                count
            },
        };
        let batch = std::mem::take(&mut self.pending_raw);
        let batch_index = self.flush_count;
        // The concatenation is rebuilt from the retained partials on failure,
        // so a rejected batch leaves the pending set intact for the retry.
        let batch_refs = batch.iter().collect::<Vec<_>>();
        let combined = match concatenate_partial_refs(execution, self.node_id, &batch_refs) {
            Ok(combined) => combined,
            Err(error) => {
                self.pending_raw = batch;
                return Err(error);
            },
        };
        let partitions = match partition_final_merge_partial(
            execution,
            self.node_id,
            batch_index,
            combined,
            &self.key_columns,
            partition_count,
        ) {
            Ok(partitions) => partitions,
            Err(error) => {
                self.pending_raw = batch;
                return Err(error);
            },
        };
        release_partials_reverse(batch);
        self.pending_raw_bytes = 0;
        for (partition_index, partition) in partitions.into_iter().enumerate() {
            if partition.frame.num_rows() == 0 {
                continue;
            }
            if self.pending[partition_index].len() >= MAX_PENDING_PARTIALS {
                self.flush_partition(execution, partition_index)?;
            }
            self.pending_bytes[partition_index] =
                self.pending_bytes[partition_index].saturating_add(partition.device_bytes());
            self.pending[partition_index].push(partition);
            self.max_pending_partials = self.max_pending_partials.max(self.pending[partition_index].len());
        }
        Ok(())
    }

    fn flush_ready_partitions(&mut self, execution: &mut BoundedExecutionAttempt<'_>) -> Result<()> {
        for partition_index in 0..self.pending.len() {
            let merged_bytes = self.merged[partition_index].iter().map(CommittedPartial::device_bytes).sum::<u64>();
            if self.pending[partition_index].len() >= MAX_PENDING_PARTIALS
                || self.pending_bytes[partition_index] >= merged_bytes.max(self.min_flush_bytes)
            {
                self.flush_partition(execution, partition_index)?;
            }
        }
        Ok(())
    }

    /// Folds one partition's pending slices into its merged state.
    ///
    /// The raw slices are merged among themselves first, then the resulting
    /// merged states are merged together, so every merge input set has one
    /// column shape. A query-cap rejection shrinks the fan-in inside the
    /// attempt's retry budget.
    fn flush_partition(&mut self, execution: &mut BoundedExecutionAttempt<'_>, partition_index: usize) -> Result<()> {
        while !self.pending[partition_index].is_empty() {
            self.merge_with_fan_in_retry(execution, partition_index, MergeInputs::PendingSlices)?;
        }
        while self.merged[partition_index].len() > 1 {
            self.merge_with_fan_in_retry(execution, partition_index, MergeInputs::MergedStates)?;
        }
        Ok(())
    }

    fn merge_with_fan_in_retry(
        &mut self,
        execution: &mut BoundedExecutionAttempt<'_>,
        partition_index: usize,
        inputs: MergeInputs,
    ) -> Result<()> {
        let available = match inputs {
            MergeInputs::PendingSlices => self.pending[partition_index].len(),
            MergeInputs::MergedStates => self.merged[partition_index].len(),
        };
        let mut fan_in = available;
        let mut attempt = 1;
        loop {
            match self.merge_prefix(execution, partition_index, inputs, fan_in.min(available)) {
                Ok(()) => return Ok(()),
                Err(error) if is_retryable_resource_error(&error) => {
                    record_retryable_resource_error(&error, execution.metrics());
                    if attempt < self.max_attempts
                        && let Some(reduced) = reduce_merge_fan_in(fan_in)
                    {
                        fan_in = reduced;
                        attempt += 1;
                        backoff_before_retry(
                            execution.context(),
                            "query_engine.exec.bounded.compiled_incremental_merge_fan_in_retry",
                        )?;
                        continue;
                    }
                    return Err(attach_incremental_merge_facts(error, partition_index, fan_in, attempt));
                },
                Err(error) => return Err(error),
            }
        }
    }

    /// Merges the first `take` inputs of one kind into a new merged state; the
    /// inputs are only released after the merged output exists.
    fn merge_prefix(
        &mut self,
        execution: &mut BoundedExecutionAttempt<'_>,
        partition_index: usize,
        inputs: MergeInputs,
        take: usize,
    ) -> Result<()> {
        let token = execution_budget_token_for_boundary(
            execution.context(),
            execution.device_budget_bytes(),
            "compiled_partitioned_final_aggregate_merge",
            "query_engine.final_merge.compiled_incremental_merge.budget_token",
            "failed to create compiled incremental partitioned final aggregate merge budget token",
        )?;
        let source = match inputs {
            MergeInputs::PendingSlices => &self.pending[partition_index],
            MergeInputs::MergedStates => &self.merged[partition_index],
        };
        let input_refs = source.iter().take(take).collect::<Vec<_>>();
        let fan_in = input_refs.len();
        let timer = Instant::now();
        let result = merge_compiled_partitioned_final_aggregate_partial_refs(
            execution,
            self.node_id,
            self.aggregate,
            self.input_schema,
            self.output_schema,
            partition_index,
            &input_refs,
            &token,
        );
        self.merge_time = self.merge_time.saturating_add(timer.elapsed());
        let frame = result?;
        let output_bytes = observe_device_memory_bytes(&frame, execution.metrics())?;
        let released = match inputs {
            MergeInputs::PendingSlices => {
                let released = self.pending[partition_index].drain(..take).collect::<Vec<_>>();
                let released_bytes = released.iter().map(CommittedPartial::device_bytes).sum::<u64>();
                self.pending_bytes[partition_index] =
                    self.pending_bytes[partition_index].saturating_sub(released_bytes);
                released
            },
            MergeInputs::MergedStates => self.merged[partition_index].drain(..take).collect::<Vec<_>>(),
        };
        release_partials_reverse(released);
        self.merged[partition_index].push(CommittedPartial::with_device_bytes(frame, output_bytes));
        self.flush_count += 1;
        self.max_partition_fan_in = self.max_partition_fan_in.max(fan_in);
        Ok(())
    }
}

struct PartitionedFinalAggregateStream {
    node_id: NodeId,
    aggregate: AggregateNode,
    input_schema: ArrowSchemaRef,
    output_schema: ArrowSchemaRef,
    partitions: VecDeque<CommittedPartial>,
}

impl ResidentFrameStreamSource for PartitionedFinalAggregateStream {
    fn alias(&self) -> Option<Box<dyn ResidentFrameStreamSource>> {
        Some(Box::new(Self {
            node_id: self.node_id,
            aggregate: self.aggregate.clone(),
            input_schema: Arc::clone(&self.input_schema),
            output_schema: Arc::clone(&self.output_schema),
            partitions: self
                .partitions
                .iter()
                .map(|partial| CommittedPartial { frame: partial.frame.clone(), device_bytes: partial.device_bytes() })
                .collect(),
        }))
    }

    fn next_frame(&mut self, execution: &mut BoundedExecutionAttempt<'_>) -> Option<Result<ResidentGpuFrame>> {
        let partial = self.partitions.pop_front()?;
        Some(
            execution
                .context()
                .check_cancellation("query_engine.aggregate.final_partition")
                .and_then(|()| {
                    execute_compiled_final_aggregate_projection(
                        execution,
                        self.node_id,
                        &self.aggregate,
                        &self.input_schema,
                        &self.output_schema,
                        partial,
                    )
                })
                .map_err(|error| error.with_fact("node_id", self.node_id)),
        )
    }

    fn retained_device_bytes(&self) -> u64 {
        self.partitions.iter().map(CommittedPartial::device_bytes).sum()
    }

    /// Every merged partition state, taken before the lazy final projection.
    /// The final merge groups by the aggregate's output key names, so these
    /// states already carry the complete distinct group-key set; only the
    /// aggregate columns still need finalization.
    fn complete_group_key_state(&self) -> Option<Vec<GpuDataFrame>> {
        Some(self.partitions.iter().map(|partial| partial.frame.clone()).collect())
    }
}

/// Which homogeneous input set one incremental merge step consumes.
#[derive(Debug, Clone, Copy)]
enum MergeInputs {
    PendingSlices,
    MergedStates,
}

fn concatenate_partial_refs(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    partials: &[&CommittedPartial],
) -> Result<CommittedPartial> {
    let cloned = partials
        .iter()
        .map(|partial| CommittedPartial { frame: partial.frame.clone(), device_bytes: partial.device_bytes() })
        .collect::<Vec<_>>();
    concatenate_partitioned_final_merge_outputs(execution, node_id, cloned)
}

fn attach_incremental_merge_facts(
    error: QueryError,
    partition_index: usize,
    fan_in: usize,
    attempt: usize,
) -> QueryError {
    error
        .with_fact_if_absent("retry_action", "fail_min_merge_fan_in")
        .with_fact_if_absent("resume_boundary", "final_merge")
        .with_fact_if_absent("partition", partition_index)
        .with_fact_if_absent("merge_fan_in", fan_in)
        .with_fact_if_absent("attempt", attempt)
}
