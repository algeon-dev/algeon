use std::collections::BTreeSet;

use super::{
    BoundedExecutionAttempt, BoundedSourceBatchingPolicy, ExecutionBudgetToken, ExecutionContext, FilterNode,
    GpuDataFrame, GpuResidentInputMap, Instant, LocalParquetFile, LocalParquetSource, MemoryControlPolicy, NodeId,
    ProjectionNode, QueryError, QueryErrorCode, QueryMetrics, ResidentGpuFrame, Result,
    SourceAdjacentAggregateChunkCapHints, SourceChunk, SourceChunkBatch, SourceChunkPlanningPolicy,
    SourceTaskMemoryModel, annotate_source_chunk_batching_reasons, apply_source_adjacent_aggregate_chunk_caps,
    chunk_retry_facts, execute_bounded_generic_local_parquet_source, execute_filter_with_budget,
    execute_projection_with_budget, execution_budget_token_for_boundary, execution_overrides, is_retryable_chunk_error,
    join_leaf_prefetch_budget_bytes, observe_device_memory_bytes,
    plan_local_parquet_source_chunks_with_execution_context, plan_source_chunk_batches, read_gpu_resident_source,
    read_literal_batch_source, record_bounded_parquet_source_metrics, record_chunk_retry_error, split_retryable_chunk,
    unsupported_bounded_shape, validate_local_parquet_source,
};
use crate::{
    dynamic_filter::{
        DynamicFilter, DynamicFilterConsumerLowering, DynamicFilterExecutionOptions, DynamicFilterId,
        DynamicFilterRoute, DynamicFilterSet,
        apply_dynamic_filters_to_local_parquet_source_with_reader_predicate_filter,
        dynamic_filter_predicate_for_schema, dynamic_filter_reader_predicate_should_merge,
    },
    exec::{
        bounded::chunks::{source_read_budget_bytes, source_read_budget_correction_from_error},
        source::{
            ScheduledLocalParquetRead, read_local_parquet_chunk_batch,
            read_local_parquet_chunk_with_reader_dynamic_filter_count, read_local_parquet_source,
        },
    },
    metrics::DynamicFilterSourceSnapshot,
};

mod batching;
mod concat;
mod dynamic_filter;
mod planning;
mod read_serial;
mod scheduler;
mod stages;
mod stream;

fn attach_join_source_read_resource_facts(
    error: QueryError,
    node_id: NodeId,
    chunk: &SourceChunk,
    attempt: usize,
    max_attempts: usize,
) -> QueryError {
    let error = error
        .with_fact_if_absent("memory_boundary", "join_source_read")
        .with_fact_if_absent("stage", "join_source_read")
        .with_fact_if_absent("node_id", node_id)
        .with_fact_if_absent("attempt", attempt)
        .with_fact_if_absent("max_attempts", max_attempts)
        .with_fact_if_absent("chunk_index", chunk.chunk_index)
        .with_fact_if_absent("file_index", chunk.file_index)
        .with_fact_if_absent("path", crate::exec::source::parquet_chunk_path_fact(chunk));
    crate::exec::source::attach_row_group_facts_if_absent(error, chunk)
}

use batching::*;
use concat::*;
use dynamic_filter::*;
pub(in crate::exec::bounded) use planning::{
    plan_bounded_aggregate_input_local_parquet_source_stream, plan_bounded_join_leaf_local_parquet_source_stream,
    plan_bounded_shared_join_aggregate_local_parquet_source_stream,
};
use read_serial::*;
use scheduler::*;
use stages::*;
use stream::*;

// Deferred Parquet leaf work that compiled join tasks can materialize in chunks.
#[derive(Clone)]
pub(in crate::exec::bounded) struct BoundedJoinLeafLocalParquetSourceStream {
    source: LocalParquetSource,
    node_id: NodeId,
    chunks: Vec<SourceChunk>,
    stages: Vec<BoundedJoinLeafSourceStreamStage>,
    reader_dynamic_filters: Vec<BoundedJoinLeafSourceStreamDynamicFilterStage>,
    planning_policy: SourceChunkPlanningPolicy,
    max_attempts: usize,
    min_retry_chunk_row_groups: usize,
}

// In-flight source read parked until its leaf resource is consumed or cached.
pub(in crate::exec::bounded) struct BoundedJoinLeafLocalParquetSourcePrefetch {
    scheduled: ScheduledLocalParquetRead,
}

// Post-decode work that stays with the leaf stream because it cannot all be pushed into Parquet.
#[derive(Clone)]
enum BoundedJoinLeafSourceStreamStage {
    Filter {
        node_id: NodeId,
        node: FilterNode,
        dynamic_filter: Option<Box<BoundedJoinLeafSourceStreamDynamicFilterStage>>,
    },
    Projection {
        node_id: NodeId,
        node: ProjectionNode,
    },
}

// Route/filter pair retained for reader-predicate and post-decode dynamic-filter metrics.
#[derive(Clone)]
struct BoundedJoinLeafSourceStreamDynamicFilterStage {
    route: DynamicFilterRoute,
    filter: DynamicFilter,
}

impl BoundedJoinLeafLocalParquetSourceStream {
    pub(in crate::exec::bounded) fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    pub(in crate::exec::bounded) fn source_node_id(&self) -> NodeId {
        self.node_id
    }

    pub(in crate::exec::bounded) fn estimated_source_read_bytes(&self) -> Option<u64> {
        self.chunks.iter().try_fold(0_u64, |total, chunk| {
            total.checked_add(
                chunk
                    .estimated_source_read_bytes
                    .or(chunk.known_row_group_bytes)
                    .or(chunk.estimated_chunk_bytes)
                    .or(chunk.known_file_bytes)?,
            )
        })
    }

    pub(in crate::exec::bounded) fn dynamic_filter_source_snapshot(&self) -> DynamicFilterSourceSnapshot {
        DynamicFilterSourceSnapshot {
            row_groups: self.chunks.iter().map(|chunk| chunk.row_group_count() as u64).sum(),
            source_bytes: self.estimated_source_read_bytes(),
        }
    }

    fn has_stages(&self) -> bool {
        !self.stages.is_empty()
    }

    pub(in crate::exec::bounded) fn with_filter_stage(mut self, node_id: NodeId, node: FilterNode) -> Self {
        self.stages.push(BoundedJoinLeafSourceStreamStage::Filter { node_id, node, dynamic_filter: None });
        self
    }

    pub(in crate::exec::bounded) fn with_projection_stage(mut self, node_id: NodeId, node: ProjectionNode) -> Self {
        self.stages.push(BoundedJoinLeafSourceStreamStage::Projection { node_id, node });
        self
    }

    pub(in crate::exec::bounded) fn deferred_reader_predicate_routes(
        &self,
        routes: &[DynamicFilterRoute],
        filter_set: &DynamicFilterSet,
        options: DynamicFilterExecutionOptions,
    ) -> BTreeSet<DynamicFilterId> {
        let routes = routes.iter().collect::<Vec<_>>();
        deferred_large_small_in_list_reader_predicate_routes(&self.source, &routes, filter_set, options)
    }

    pub(in crate::exec::bounded) fn with_dynamic_filters_and_deferred_reader_predicates(
        mut self,
        context: &ExecutionContext,
        routes: &[&DynamicFilterRoute],
        filter_set: &DynamicFilterSet,
        deferred_reader_predicate_routes: &BTreeSet<DynamicFilterId>,
        options: DynamicFilterExecutionOptions,
    ) -> Result<Self> {
        if routes.is_empty() {
            return Ok(self);
        }
        let merged_reader_predicate_routes = merged_reader_predicate_route_ids(
            &self.source,
            routes,
            filter_set,
            deferred_reader_predicate_routes,
            options,
        )?;
        let before = self.dynamic_filter_source_snapshot();
        let source = apply_dynamic_filters_to_local_parquet_source_with_reader_predicate_filter(
            &self.source,
            routes,
            filter_set,
            |route, _filter| merged_reader_predicate_routes.contains(&route.id),
        )?;
        let mut chunk_plan = plan_local_parquet_source_chunks_with_execution_context(
            self.node_id,
            &source,
            self.planning_policy,
            context,
        )?;
        annotate_join_leaf_source_chunk_batching_reasons(
            &source,
            &mut chunk_plan.chunks,
            self.planning_policy.max_source_chunk_bytes,
        );
        let after = DynamicFilterSourceSnapshot {
            row_groups: chunk_plan.chunks.iter().map(|chunk| chunk.row_group_count() as u64).sum(),
            source_bytes: chunk_plan.chunks.iter().try_fold(0_u64, |total, chunk| {
                total.checked_add(
                    chunk
                        .estimated_source_read_bytes
                        .or(chunk.known_row_group_bytes)
                        .or(chunk.estimated_chunk_bytes)
                        .or(chunk.known_file_bytes)?,
                )
            }),
        };
        let post_decode_filters = source_post_decode_dynamic_filter_stages(
            &source,
            self.node_id,
            routes,
            filter_set,
            before,
            after,
            deferred_reader_predicate_routes,
            options,
        );
        let reader_dynamic_filters =
            source_reader_predicate_dynamic_filter_stages(routes, filter_set, &merged_reader_predicate_routes);
        self.source = source;
        self.chunks = chunk_plan.chunks;
        self.reader_dynamic_filters.extend(reader_dynamic_filters);
        if !post_decode_filters.is_empty() {
            let mut stages = post_decode_filters
                .iter()
                .map(|(route, filter, node)| BoundedJoinLeafSourceStreamStage::Filter {
                    node_id: route.consumer.node_id(),
                    node: node.clone(),
                    dynamic_filter: Some(Box::new(BoundedJoinLeafSourceStreamDynamicFilterStage {
                        route: route.clone(),
                        filter: filter.clone(),
                    })),
                })
                .collect::<Vec<_>>();
            stages.extend(self.stages);
            self.stages = stages;
        }
        Ok(self)
    }
}

impl BoundedJoinLeafLocalParquetSourcePrefetch {
    pub(in crate::exec::bounded) fn reservation_bytes(&self) -> u64 {
        self.scheduled.budget_bytes()
    }

    pub(in crate::exec::bounded) fn finish(
        self,
        execution: &mut BoundedExecutionAttempt<'_>,
    ) -> Result<ResidentGpuFrame> {
        let is_prefetch = self.scheduled.is_prefetch();
        let output = self.scheduled.finish(execution)?;
        if is_prefetch {
            execution.metrics_mut().record_source_read_prefetch_completed();
        }
        let device_bytes = observe_device_memory_bytes(&output.frame, execution.metrics_mut())?;
        execution.metrics_mut().record_source_chunk_commit(
            output
                .observed_row_group_count
                .or_else(|| output.chunk.known_read_row_group_count().map(|row_groups| row_groups as u64))
                .unwrap_or_else(|| output.chunk.row_group_count() as u64),
        );
        Ok(ResidentGpuFrame::with_device_bytes(output.frame, device_bytes))
    }

    pub(in crate::exec::bounded) fn abandon(self, execution: &mut BoundedExecutionAttempt<'_>) {
        self.scheduled.abandon(execution);
    }
}

pub(in crate::exec::bounded) fn try_start_bounded_join_leaf_local_parquet_source_prefetch(
    execution: &mut BoundedExecutionAttempt<'_>,
    stream: &BoundedJoinLeafLocalParquetSourceStream,
) -> Result<Option<BoundedJoinLeafLocalParquetSourcePrefetch>> {
    if !execution.context().source_scheduler_policy().enables_prefetch()
        || stream.has_stages()
        || !stream.reader_dynamic_filters.is_empty()
        || stream.chunks.len() != 1
        || !join_leaf_source_prefetch_remote_reader_admitted(&stream.source)
    {
        return Ok(None);
    }
    let chunk = stream.chunks.first().expect("single chunk checked");
    let requested_bytes = source_read_budget_bytes(execution.context().memory_control_policy(), chunk);
    execution.metrics_mut().record_source_read_prefetch_started();
    let scheduled = start_join_leaf_scheduled_source_read(
        execution,
        &stream.source,
        chunk,
        requested_bytes,
        "join_source_read_prefetch",
        false,
        true,
        1,
        stream.max_attempts,
    )?;
    Ok(Some(BoundedJoinLeafLocalParquetSourcePrefetch { scheduled }))
}

fn join_leaf_source_prefetch_remote_reader_admitted(source: &LocalParquetSource) -> bool {
    !source.files.is_empty()
        && source.files.iter().all(|file| {
            matches!(file.resolved_source.read_strategy(), crate::source::ParquetReadStrategy::RemoteDirect)
        })
}

pub(in crate::exec::bounded) fn execute_bounded_join_leaf_source(
    execution: &mut BoundedExecutionAttempt<'_>,
    source: &crate::source::SourceNode,
    node_id: NodeId,
    gpu_inputs: &GpuResidentInputMap,
) -> Result<ResidentGpuFrame> {
    match source {
        crate::source::SourceNode::LocalParquet(source) => {
            validate_local_parquet_source(source)?;
            if source.has_position_delete_visible_scan_limit() {
                return execute_bounded_generic_local_parquet_source(execution, source, node_id);
            }
            execute_bounded_join_leaf_local_parquet_source(execution, source, node_id)
        },
        crate::source::SourceNode::GpuResident(source) => {
            let frame = read_gpu_resident_source(execution.context(), source, gpu_inputs, execution.metrics_mut())?;
            let device_bytes = observe_device_memory_bytes(&frame, execution.metrics_mut())?;
            Ok(ResidentGpuFrame::with_device_bytes(frame, device_bytes))
        },
        crate::source::SourceNode::LiteralBatch(source) => {
            let frame = read_literal_batch_source(execution.context(), source, execution.metrics_mut())?;
            let device_bytes = observe_device_memory_bytes(&frame, execution.metrics_mut())?;
            Ok(ResidentGpuFrame::with_device_bytes(frame, device_bytes))
        },
    }
}
pub(in crate::exec::bounded) fn materialize_bounded_join_leaf_local_parquet_source_stream(
    execution: &mut BoundedExecutionAttempt<'_>,
    stream: BoundedJoinLeafLocalParquetSourceStream,
) -> Result<ResidentGpuFrame> {
    let node_id = stream.node_id;
    if stream.chunks.is_empty() {
        // Every row group was pruned before execution; the streaming paths
        // below would otherwise yield zero chunks and no frame to return.
        let mut empty = empty_join_leaf_source_frame(execution, &stream)?;
        if !stream.stages.is_empty() {
            empty = apply_join_leaf_source_stream_stages(execution, &stream, empty)?;
        }
        return Ok(empty);
    }
    if stream.has_stages() {
        let mut chunks = Vec::with_capacity(stream.chunk_count());
        let stream_result = stream_join_leaf_source_chunks_serial(execution, &stream, &mut |input, _| {
            let device_bytes = input.device_bytes();
            chunks.push((input.frame, device_bytes));
            Ok(())
        });
        if let Err(error) = stream_result {
            release_join_source_chunks_reverse(chunks);
            return Err(error);
        }
        return concatenate_join_source_chunks(execution, node_id, chunks);
    }

    let chunks = read_join_leaf_source_stream_chunks(execution, &stream)?;
    concatenate_join_source_chunks(execution, node_id, chunks)
}

pub(in crate::exec::bounded) fn stream_bounded_join_leaf_local_parquet_source<F>(
    execution: &mut BoundedExecutionAttempt<'_>,
    stream: BoundedJoinLeafLocalParquetSourceStream,
    mut consume: F,
) -> Result<()>
where
    F: FnMut(ResidentGpuFrame, &mut BoundedExecutionAttempt<'_>) -> Result<()>,
{
    if stream_has_multi_source_batches(execution.context(), &stream) {
        stream_join_leaf_source_chunks_serial(execution, &stream, &mut consume)
    } else if execution.context().source_scheduler_policy().enables_prefetch() && stream.chunks.len() > 1 {
        stream_join_leaf_source_chunks_with_scheduler(execution, &stream, &mut consume)
    } else {
        stream_join_leaf_source_chunks_serial(execution, &stream, &mut consume)
    }
}

pub(in crate::exec::bounded) fn stream_bounded_join_leaf_local_parquet_source_chunks<F>(
    execution: &mut BoundedExecutionAttempt<'_>,
    stream: BoundedJoinLeafLocalParquetSourceStream,
    mut consume: F,
) -> Result<()>
where
    F: FnMut(ResidentGpuFrame, &mut BoundedExecutionAttempt<'_>) -> Result<()>,
{
    for chunk in stream.chunks.clone() {
        stream_join_leaf_source_chunk_with_consumer_retry(execution, &stream, chunk, 1, &mut consume)?;
    }
    Ok(())
}

pub(in crate::exec::bounded) fn stream_bounded_join_leaf_local_parquet_source_serial<F>(
    execution: &mut BoundedExecutionAttempt<'_>,
    stream: BoundedJoinLeafLocalParquetSourceStream,
    mut consume: F,
) -> Result<()>
where
    F: FnMut(ResidentGpuFrame, &mut BoundedExecutionAttempt<'_>) -> Result<()>,
{
    stream_join_leaf_source_chunks_serial(execution, &stream, &mut consume)
}

fn execute_bounded_join_leaf_local_parquet_source(
    execution: &mut BoundedExecutionAttempt<'_>,
    source: &LocalParquetSource,
    node_id: NodeId,
) -> Result<ResidentGpuFrame> {
    let stream = plan_bounded_join_leaf_local_parquet_source_stream(
        execution.context(),
        source,
        node_id,
        false,
        execution.metrics_mut(),
    )?;
    let chunks = read_join_leaf_source_stream_chunks(execution, &stream)?;
    concatenate_join_source_chunks(execution, node_id, chunks)
}

fn read_join_leaf_source_stream_chunks(
    execution: &mut BoundedExecutionAttempt<'_>,
    stream: &BoundedJoinLeafLocalParquetSourceStream,
) -> Result<Vec<(GpuDataFrame, u64)>> {
    if stream.chunks.is_empty() {
        let mut empty = empty_join_leaf_source_frame(execution, stream)?;
        if !stream.stages.is_empty() {
            empty = apply_join_leaf_source_stream_stages(execution, stream, empty)?;
        }
        let device_bytes = empty.device_bytes();
        return Ok(vec![(empty.frame, device_bytes)]);
    }

    let materialized_batches = materialized_source_chunk_batches_for_execution(stream);
    let mut chunks = if materialized_batches.iter().any(|batch| batch.chunk_count() > 1) {
        read_join_leaf_source_batches_serial(
            execution,
            &stream.source,
            stream.node_id,
            materialized_batches,
            &stream.reader_dynamic_filters,
            stream.max_attempts,
            stream.min_retry_chunk_row_groups,
        )?
    } else if execution.context().source_scheduler_policy().enables_prefetch() && stream.chunks.len() > 1 {
        match read_join_leaf_source_chunks_with_scheduler(
            execution,
            &stream.source,
            stream.node_id,
            &stream.chunks,
            &stream.reader_dynamic_filters,
            stream.max_attempts,
        ) {
            Ok(chunks) => chunks,
            Err(error) if is_retryable_chunk_error(&error) => read_join_leaf_source_chunks_serial(
                execution,
                &stream.source,
                stream.node_id,
                &stream.chunks,
                &stream.reader_dynamic_filters,
                stream.max_attempts,
                stream.min_retry_chunk_row_groups,
            )?,
            Err(error) => return Err(error),
        }
    } else {
        read_join_leaf_source_chunks_serial(
            execution,
            &stream.source,
            stream.node_id,
            &stream.chunks,
            &stream.reader_dynamic_filters,
            stream.max_attempts,
            stream.min_retry_chunk_row_groups,
        )?
    };
    if !stream.stages.is_empty() {
        let chunk_count = chunks.len();
        let mut pending = chunks;
        let mut transformed = Vec::with_capacity(chunk_count);
        while !pending.is_empty() {
            let (frame, reservation) = pending.remove(0);
            let output = apply_join_leaf_source_stream_stages(
                execution,
                stream,
                ResidentGpuFrame::with_device_bytes(frame, reservation),
            );
            match output {
                Ok(output) => {
                    let device_bytes = output.device_bytes();
                    transformed.push((output.frame, device_bytes));
                },
                Err(error) => {
                    release_join_source_chunks_reverse(pending);
                    release_join_source_chunks_reverse(transformed);
                    return Err(error);
                },
            }
        }
        chunks = transformed;
    }
    Ok(chunks)
}

fn empty_join_leaf_source_frame(
    execution: &mut BoundedExecutionAttempt<'_>,
    stream: &BoundedJoinLeafLocalParquetSourceStream,
) -> Result<ResidentGpuFrame> {
    let frame =
        read_local_parquet_source(execution.context(), stream.node_id, &stream.source, execution.metrics_mut())?;
    let device_bytes = observe_device_memory_bytes(&frame, execution.metrics_mut())?;
    Ok(ResidentGpuFrame::with_device_bytes(frame, device_bytes))
}
