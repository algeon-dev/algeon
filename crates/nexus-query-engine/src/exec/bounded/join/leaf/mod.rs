use std::collections::{BTreeSet, VecDeque};

use super::*;
use crate::{
    dynamic_filter::{
        DynamicFilter, DynamicFilterConsumerLowering, DynamicFilterId, DynamicFilterRoute, DynamicFilterSet,
        apply_dynamic_filters_to_local_parquet_source_with_reader_predicate_filter,
        dynamic_filter_predicate_for_schema, dynamic_filter_reader_predicate_should_merge,
    },
    exec::{
        bounded::chunks::{
            SourceReadReservationContractQuality, insert_bounded_local_parquet_source_cache,
            source_read_budget_correction_from_error, source_read_reservation_contract,
            try_grow_source_read_reservation_for_budget_correction, try_read_cached_local_parquet_chunk,
        },
        source::{
            ScheduledLocalParquetRead, format_row_groups, read_local_parquet_chunk_batch,
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
mod reservation;
mod scheduler;
mod stages;
mod stream;

use batching::*;
use concat::*;
use dynamic_filter::*;
pub(in crate::exec::bounded) use planning::plan_bounded_join_leaf_local_parquet_source_stream;
use read_serial::*;
use reservation::*;
use scheduler::*;
use stages::*;
use stream::*;

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
    ) -> BTreeSet<DynamicFilterId> {
        let routes = routes.iter().collect::<Vec<_>>();
        deferred_large_small_in_list_reader_predicate_routes(&self.source, &routes, filter_set)
    }

    pub(in crate::exec::bounded) fn with_dynamic_filters_and_deferred_reader_predicates(
        mut self,
        routes: &[&DynamicFilterRoute],
        filter_set: &DynamicFilterSet,
        deferred_reader_predicate_routes: &BTreeSet<DynamicFilterId>,
    ) -> Result<Self> {
        if routes.is_empty() {
            return Ok(self);
        }
        let merged_reader_predicate_routes =
            merged_reader_predicate_route_ids(&self.source, routes, filter_set, deferred_reader_predicate_routes)?;
        let before = self.dynamic_filter_source_snapshot();
        let source = apply_dynamic_filters_to_local_parquet_source_with_reader_predicate_filter(
            &self.source,
            routes,
            filter_set,
            |route, _filter| merged_reader_predicate_routes.contains(&route.id),
        )?;
        let mut chunk_plan = plan_local_parquet_source_chunks(self.node_id, &source, self.planning_policy)?;
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

pub(in crate::exec::bounded) fn execute_bounded_join_leaf_source(
    context: &ExecutionContext,
    source: &crate::source::SourceNode,
    node_id: NodeId,
    gpu_inputs: &GpuResidentInputMap,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    match source {
        crate::source::SourceNode::LocalParquet(source) => {
            validate_local_parquet_source(source)?;
            execute_bounded_join_leaf_local_parquet_source(context, source, node_id, reservations, metrics)
        },
        crate::source::SourceNode::GpuResident(source) => {
            let frame = read_gpu_resident_source(context, source, gpu_inputs, metrics)?;
            let requested_bytes = observe_device_memory_bytes(&frame, metrics)?;
            let reservation = reservations.try_reserve(
                MemoryReservationRequest::new(ReservationKind::SourceRead, requested_bytes, reservations.budget())
                    .with_memory_boundary("gpu_resident_join_input")
                    .with_node_id(node_id)
                    .with_stage("gpu_resident_join_input")
                    .with_attempt(1, 1),
                metrics,
            )?;
            Ok(ResidentGpuFrame { frame, reservation })
        },
        crate::source::SourceNode::LiteralBatch(source) => {
            let frame = read_literal_batch_source(context, source, metrics)?;
            let requested_bytes = observe_device_memory_bytes(&frame, metrics)?;
            let reservation = reservations.try_reserve(
                MemoryReservationRequest::new(ReservationKind::SourceRead, requested_bytes, reservations.budget())
                    .with_memory_boundary("literal_batch_join_input")
                    .with_node_id(node_id)
                    .with_stage("literal_batch_join_input")
                    .with_attempt(1, 1),
                metrics,
            )?;
            Ok(ResidentGpuFrame { frame, reservation })
        },
    }
}
pub(in crate::exec::bounded) fn materialize_bounded_join_leaf_local_parquet_source_stream(
    context: &ExecutionContext,
    stream: BoundedJoinLeafLocalParquetSourceStream,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    let node_id = stream.node_id;
    if stream.has_stages() {
        let mut chunks = Vec::with_capacity(stream.chunk_count());
        let stream_result =
            stream_join_leaf_source_chunks_serial(context, &stream, reservations, metrics, &mut |input, _, _| {
                chunks.push((input.frame, input.reservation));
                Ok(())
            });
        if let Err(error) = stream_result {
            release_join_source_chunks_reverse(reservations, chunks, metrics);
            return Err(error);
        }
        return concatenate_join_source_chunks(context, node_id, chunks, reservations, metrics);
    }

    let chunks = read_join_leaf_source_stream_chunks(context, &stream, reservations, metrics)?;
    concatenate_join_source_chunks(context, node_id, chunks, reservations, metrics)
}

pub(in crate::exec::bounded) fn stream_bounded_join_leaf_local_parquet_source<F>(
    context: &ExecutionContext,
    stream: BoundedJoinLeafLocalParquetSourceStream,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    mut consume: F,
) -> Result<()>
where
    F: FnMut(ResidentGpuFrame, &mut ReservationTracker, &mut QueryMetrics) -> Result<()>,
{
    if stream_has_multi_source_batches(&stream) {
        stream_join_leaf_source_chunks_serial(context, &stream, reservations, metrics, &mut consume)
    } else if context.source_scheduler_policy().enables_prefetch() && stream.chunks.len() > 1 {
        stream_join_leaf_source_chunks_with_scheduler(context, &stream, reservations, metrics, &mut consume)
    } else {
        stream_join_leaf_source_chunks_serial(context, &stream, reservations, metrics, &mut consume)
    }
}

pub(in crate::exec::bounded) fn stream_bounded_join_leaf_local_parquet_source_serial<F>(
    context: &ExecutionContext,
    stream: BoundedJoinLeafLocalParquetSourceStream,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    mut consume: F,
) -> Result<()>
where
    F: FnMut(ResidentGpuFrame, &mut ReservationTracker, &mut QueryMetrics) -> Result<()>,
{
    stream_join_leaf_source_chunks_serial(context, &stream, reservations, metrics, &mut consume)
}

fn execute_bounded_join_leaf_local_parquet_source(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    node_id: NodeId,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    let stream = plan_bounded_join_leaf_local_parquet_source_stream(context, source, node_id, false, metrics)?;
    let chunks = read_join_leaf_source_stream_chunks(context, &stream, reservations, metrics)?;
    concatenate_join_source_chunks(context, node_id, chunks, reservations, metrics)
}

fn read_join_leaf_source_stream_chunks(
    context: &ExecutionContext,
    stream: &BoundedJoinLeafLocalParquetSourceStream,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<Vec<(GpuDataFrame, ActiveReservation)>> {
    if stream.chunks.is_empty() {
        let mut empty = empty_join_leaf_source_frame(context, stream, reservations, metrics)?;
        if !stream.stages.is_empty() {
            empty = apply_join_leaf_source_stream_stages(context, stream, empty, reservations, metrics)?;
        }
        return Ok(vec![(empty.frame, empty.reservation)]);
    }

    let grow_source_to_observed = stream.stages.is_empty();
    let mut chunks = if stream_has_multi_source_batches(stream) {
        read_join_leaf_source_chunks_serial(
            context,
            &stream.source,
            stream.node_id,
            &stream.chunks,
            &stream.reader_dynamic_filters,
            reservations,
            metrics,
            stream.max_attempts,
            stream.min_retry_chunk_row_groups,
            grow_source_to_observed,
        )?
    } else if context.source_scheduler_policy().enables_prefetch() && stream.chunks.len() > 1 {
        match read_join_leaf_source_chunks_with_scheduler(
            context,
            &stream.source,
            stream.node_id,
            &stream.chunks,
            &stream.reader_dynamic_filters,
            reservations,
            metrics,
            stream.max_attempts,
            grow_source_to_observed,
        ) {
            Ok(chunks) => chunks,
            Err(error) if is_retryable_chunk_error(&error) => read_join_leaf_source_chunks_serial(
                context,
                &stream.source,
                stream.node_id,
                &stream.chunks,
                &stream.reader_dynamic_filters,
                reservations,
                metrics,
                stream.max_attempts,
                stream.min_retry_chunk_row_groups,
                grow_source_to_observed,
            )?,
            Err(error) => return Err(error),
        }
    } else {
        read_join_leaf_source_chunks_serial(
            context,
            &stream.source,
            stream.node_id,
            &stream.chunks,
            &stream.reader_dynamic_filters,
            reservations,
            metrics,
            stream.max_attempts,
            stream.min_retry_chunk_row_groups,
            grow_source_to_observed,
        )?
    };
    if !stream.stages.is_empty() {
        let chunk_count = chunks.len();
        let mut pending = chunks;
        let mut transformed = Vec::with_capacity(chunk_count);
        while !pending.is_empty() {
            let (frame, reservation) = pending.remove(0);
            let output = apply_join_leaf_source_stream_stages(
                context,
                stream,
                ResidentGpuFrame { frame, reservation },
                reservations,
                metrics,
            );
            match output {
                Ok(output) => transformed.push((output.frame, output.reservation)),
                Err(error) => {
                    release_join_source_chunks_reverse(reservations, pending, metrics);
                    release_join_source_chunks_reverse(reservations, transformed, metrics);
                    return Err(error);
                },
            }
        }
        chunks = transformed;
    }
    Ok(chunks)
}

fn empty_join_leaf_source_frame(
    context: &ExecutionContext,
    stream: &BoundedJoinLeafLocalParquetSourceStream,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    let frame = read_local_parquet_source(context, &stream.source, metrics)?;
    let requested_bytes = observe_device_memory_bytes(&frame, metrics)?;
    let reservation = reservations.try_reserve(
        MemoryReservationRequest::new(ReservationKind::SourceRead, requested_bytes, reservations.budget())
            .with_memory_boundary("join_source_empty_output")
            .with_node_id(stream.node_id)
            .with_stage("join_source_empty_output")
            .with_attempt(1, 1),
        metrics,
    )?;
    Ok(ResidentGpuFrame { frame, reservation })
}
