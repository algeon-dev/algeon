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

const MAX_UNPRUNED_HASH_SET_POST_DECODE_VALUES: usize = 4096;
const MIN_UNPRUNED_HASH_SET_POST_DECODE_SOURCE_ROWS: u64 = 1_000_000;
const MAX_DEFERRED_SMALL_IN_LIST_READER_VALUES: usize = 128;

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

fn merged_reader_predicate_route_ids(
    source: &LocalParquetSource,
    routes: &[&DynamicFilterRoute],
    filter_set: &DynamicFilterSet,
    deferred_reader_predicate_routes: &BTreeSet<DynamicFilterId>,
) -> Result<BTreeSet<DynamicFilterId>> {
    let mut ids = BTreeSet::new();
    for route in routes {
        if deferred_reader_predicate_routes.contains(&route.id) {
            continue;
        }
        if !matches!(
            route.consumer_lowering,
            DynamicFilterConsumerLowering::ReaderPredicate | DynamicFilterConsumerLowering::MergedReaderPredicate
        ) {
            continue;
        }
        let Some(filter) = filter_set.produced(route.id) else {
            continue;
        };
        let Some(column) = route.consumer_columns.first() else {
            continue;
        };
        if dynamic_filter_reader_predicate_should_merge(source, column, filter)? {
            ids.insert(route.id);
        }
    }
    Ok(ids)
}

fn source_reader_predicate_dynamic_filter_stages(
    routes: &[&DynamicFilterRoute],
    filter_set: &DynamicFilterSet,
    merged_reader_predicate_routes: &BTreeSet<DynamicFilterId>,
) -> Vec<BoundedJoinLeafSourceStreamDynamicFilterStage> {
    routes
        .iter()
        .filter_map(|route| {
            if !merged_reader_predicate_routes.contains(&route.id) {
                return None;
            }
            let filter = filter_set.produced(route.id)?;
            if matches!(filter, DynamicFilter::Empty | DynamicFilter::HashSetContainsI64 { .. }) {
                return None;
            }
            Some(BoundedJoinLeafSourceStreamDynamicFilterStage { route: (*route).clone(), filter: filter.clone() })
        })
        .collect()
}

fn deferred_large_small_in_list_reader_predicate_routes(
    source: &LocalParquetSource,
    routes: &[&DynamicFilterRoute],
    filter_set: &DynamicFilterSet,
) -> BTreeSet<DynamicFilterId> {
    let mut large_small_in_list_routes = BTreeSet::new();
    let mut retained_reader_predicate_count = usize::from(source.reader_predicate.is_some());
    for route in routes {
        if !matches!(
            route.consumer_lowering,
            DynamicFilterConsumerLowering::ReaderPredicate | DynamicFilterConsumerLowering::MergedReaderPredicate
        ) {
            continue;
        }
        let Some(filter) = filter_set.produced(route.id) else {
            continue;
        };
        if large_small_in_list_reader_predicate_should_defer(filter) {
            large_small_in_list_routes.insert(route.id);
        } else if !matches!(filter, DynamicFilter::Empty | DynamicFilter::HashSetContainsI64 { .. }) {
            retained_reader_predicate_count = retained_reader_predicate_count.saturating_add(1);
        }
    }
    if retained_reader_predicate_count == 0 {
        return BTreeSet::new();
    }
    large_small_in_list_routes
}

fn large_small_in_list_reader_predicate_should_defer(filter: &DynamicFilter) -> bool {
    matches!(
        filter,
        DynamicFilter::SmallInListI64 {
            values,
            ..
        } if values.len() > MAX_DEFERRED_SMALL_IN_LIST_READER_VALUES
    )
}

fn source_post_decode_dynamic_filter_stages(
    source: &LocalParquetSource,
    source_node_id: NodeId,
    routes: &[&DynamicFilterRoute],
    filter_set: &DynamicFilterSet,
    before: DynamicFilterSourceSnapshot,
    after: DynamicFilterSourceSnapshot,
    deferred_reader_predicate_routes: &BTreeSet<DynamicFilterId>,
) -> Vec<(DynamicFilterRoute, DynamicFilter, FilterNode)> {
    routes
        .iter()
        .filter_map(|route| {
            let filter = filter_set.produced(route.id)?;
            if !source_dynamic_filter_requires_post_decode(
                source,
                route,
                filter,
                before,
                after,
                deferred_reader_predicate_routes,
            ) {
                return None;
            }
            let column = route.consumer_columns.first()?;
            let predicate = dynamic_filter_predicate_for_schema(source.schema.as_ref(), column, filter)?;
            Some(((*route).clone(), filter.clone(), FilterNode { input: source_node_id, predicate }))
        })
        .collect()
}

fn source_dynamic_filter_requires_post_decode(
    source: &LocalParquetSource,
    route: &DynamicFilterRoute,
    filter: &DynamicFilter,
    before: DynamicFilterSourceSnapshot,
    after: DynamicFilterSourceSnapshot,
    deferred_reader_predicate_routes: &BTreeSet<DynamicFilterId>,
) -> bool {
    !matches!(filter, DynamicFilter::Empty)
        && (deferred_reader_predicate_routes.contains(&route.id)
            || unpruned_reader_hash_set_should_run_post_decode(source, route, filter, before, after)
            || matches!(
                route.consumer_lowering,
                DynamicFilterConsumerLowering::PostDecodeOnly
                    | DynamicFilterConsumerLowering::SkippedUnsupportedStaticPredicateMix
            ))
}

fn unpruned_reader_hash_set_should_run_post_decode(
    source: &LocalParquetSource,
    route: &DynamicFilterRoute,
    filter: &DynamicFilter,
    before: DynamicFilterSourceSnapshot,
    after: DynamicFilterSourceSnapshot,
) -> bool {
    if route.consumer_lowering != DynamicFilterConsumerLowering::ReaderPredicate {
        return false;
    }
    if !dynamic_filter_source_pruning_unchanged(before, after) {
        return false;
    }
    let DynamicFilter::HashSetContainsI64 { values, .. } = filter else {
        return false;
    };
    if values.len() > MAX_UNPRUNED_HASH_SET_POST_DECODE_VALUES {
        return false;
    }
    selected_source_known_rows(source).is_some_and(|rows| rows >= MIN_UNPRUNED_HASH_SET_POST_DECODE_SOURCE_ROWS)
}

fn dynamic_filter_source_pruning_unchanged(
    before: DynamicFilterSourceSnapshot,
    after: DynamicFilterSourceSnapshot,
) -> bool {
    before.row_groups == after.row_groups && before.source_bytes == after.source_bytes
}

fn selected_source_known_rows(source: &LocalParquetSource) -> Option<u64> {
    source.files.iter().try_fold(0_u64, |total, file| {
        let row_group_rows = file.row_group_rows.as_ref()?;
        let selected_rows = match &file.row_groups {
            Some(row_groups) => row_groups.iter().try_fold(0_u64, |sum, row_group| {
                row_group_rows.get(*row_group).and_then(|rows| sum.checked_add(*rows))
            })?,
            None => row_group_rows.iter().try_fold(0_u64, |sum, rows| sum.checked_add(*rows))?,
        };
        total.checked_add(selected_rows)
    })
}

fn record_dynamic_filter_reader_predicate_rows(
    reader_dynamic_filters: &[BoundedJoinLeafSourceStreamDynamicFilterStage],
    chunk: &SourceChunk,
    frame: &GpuDataFrame,
    metrics: &mut QueryMetrics,
) {
    if reader_dynamic_filters.is_empty() {
        return;
    }
    let rows_before = chunk.known_row_count;
    let rows_after = Some(frame.num_rows() as u64);
    metrics.record_dynamic_filter_reader_predicate_source_reduction(rows_before, rows_after);
    for dynamic_filter in reader_dynamic_filters {
        metrics.record_dynamic_filter_source_reader_predicate_rows(
            &dynamic_filter.route,
            &dynamic_filter.filter,
            chunk.file_index,
            chunk.chunk_index,
            rows_before,
            rows_after,
        );
    }
}

fn record_dynamic_filter_reader_predicate_batch_rows(
    reader_dynamic_filters: &[BoundedJoinLeafSourceStreamDynamicFilterStage],
    batch: &SourceChunkBatch,
    frame: &GpuDataFrame,
    metrics: &mut QueryMetrics,
) {
    if reader_dynamic_filters.is_empty() {
        return;
    }
    let rows_before =
        batch.chunks.iter().try_fold(0_u64, |total, chunk| Some(total.saturating_add(chunk.known_row_count?)));
    let rows_after = Some(frame.num_rows() as u64);
    metrics.record_dynamic_filter_reader_predicate_source_reduction(rows_before, rows_after);
    for dynamic_filter in reader_dynamic_filters {
        metrics.record_dynamic_filter_source_reader_predicate_batch_rows(
            &dynamic_filter.route,
            &dynamic_filter.filter,
            rows_before,
            rows_after,
        );
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

pub(in crate::exec::bounded) fn plan_bounded_join_leaf_local_parquet_source_stream(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    node_id: NodeId,
    has_source_adjacent_stages: bool,
    metrics: &mut QueryMetrics,
) -> Result<BoundedJoinLeafLocalParquetSourceStream> {
    validate_local_parquet_source(source)?;
    let planning_policy = join_source_chunk_planning_policy(context, has_source_adjacent_stages);
    let mut chunk_plan = plan_local_parquet_source_chunks(node_id, source, planning_policy)?;
    annotate_join_leaf_source_chunk_batching_reasons(
        source,
        &mut chunk_plan.chunks,
        planning_policy.max_source_chunk_bytes,
    );
    for chunk in &chunk_plan.chunks {
        metrics.record_source_chunk_planned(chunk.row_group_count() as u64);
    }
    if chunk_plan.chunks.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.bounded.join.source",
            "bounded join leaf local parquet source requires at least one planned source chunk",
        )
        .with_fact("source_node_id", node_id));
    }
    record_bounded_parquet_source_metrics(source, &chunk_plan, metrics);
    let (max_attempts, min_retry_chunk_row_groups) = join_source_retry_policy(context);
    Ok(BoundedJoinLeafLocalParquetSourceStream {
        source: source.clone(),
        node_id,
        chunks: chunk_plan.chunks,
        stages: Vec::new(),
        reader_dynamic_filters: Vec::new(),
        planning_policy,
        max_attempts,
        min_retry_chunk_row_groups,
    })
}

fn join_source_chunk_planning_policy(
    context: &ExecutionContext,
    has_source_adjacent_stages: bool,
) -> SourceChunkPlanningPolicy {
    SourceChunkPlanningPolicy::for_source_task(
        context.memory_control_policy(),
        SourceTaskMemoryModel::join_leaf_stream(has_source_adjacent_stages),
    )
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

fn stream_has_multi_source_batches(stream: &BoundedJoinLeafLocalParquetSourceStream) -> bool {
    stream.source.partition_columns.is_empty()
        && source_chunk_batching_reader_predicate_supported(&stream.source)
        && source_chunk_batches(stream).iter().any(|batch| batch.chunk_count() > 1)
}

fn source_chunk_batches(stream: &BoundedJoinLeafLocalParquetSourceStream) -> Vec<SourceChunkBatch> {
    join_leaf_source_chunk_batches(
        &stream.source,
        &stream.chunks,
        &stream.reader_dynamic_filters,
        stream.planning_policy.max_source_chunk_bytes,
    )
}

fn join_leaf_source_chunk_batches(
    source: &LocalParquetSource,
    chunks: &[SourceChunk],
    _reader_dynamic_filters: &[BoundedJoinLeafSourceStreamDynamicFilterStage],
    max_batch_source_read_bytes: Option<u64>,
) -> Vec<SourceChunkBatch> {
    if let Some(reason) = join_leaf_source_batching_disabled_reason(source) {
        return singleton_source_chunk_batches(chunks, reason);
    }
    let target_bytes = if source_chunk_batch_decode_scratch_risk(source, chunks) {
        capped_source_chunk_batch_target_bytes(max_batch_source_read_bytes)
    } else {
        max_batch_source_read_bytes
    };
    plan_source_chunk_batches(chunks, target_bytes)
}

fn source_chunk_batching_reader_predicate_supported(source: &LocalParquetSource) -> bool {
    source.reader_predicate.as_ref().is_none_or(|predicate| source.reader_predicate_is_supported(predicate))
}

fn join_leaf_source_batching_disabled_reason(source: &LocalParquetSource) -> Option<&'static str> {
    if !source_chunk_batching_reader_predicate_supported(source) {
        return Some("source_gate:unsupported_reader_predicate");
    }
    if !source.partition_columns.is_empty() {
        return Some("source_gate:partition_columns");
    }
    None
}

fn annotate_join_leaf_source_chunk_batching_reasons(
    source: &LocalParquetSource,
    chunks: &mut [SourceChunk],
    max_batch_source_read_bytes: Option<u64>,
) {
    let target_bytes = if source_chunk_batch_decode_scratch_risk(source, chunks) {
        capped_source_chunk_batch_target_bytes(max_batch_source_read_bytes)
    } else {
        max_batch_source_read_bytes
    };
    annotate_source_chunk_batching_reasons(chunks, join_leaf_source_batching_disabled_reason(source), target_bytes);
}

fn singleton_source_chunk_batches(chunks: &[SourceChunk], reason: &'static str) -> Vec<SourceChunkBatch> {
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
            source_batching_reason: reason.to_string(),
        })
        .collect()
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

fn join_source_retry_policy(context: &ExecutionContext) -> (usize, usize) {
    match context.memory_control_policy() {
        MemoryControlPolicy::Bounded { max_retry_attempts, min_retry_chunk_row_groups, .. } => {
            (max_retry_attempts.max(1), min_retry_chunk_row_groups.max(1))
        },
        MemoryControlPolicy::Unbounded => (1, 1),
    }
}
fn read_join_leaf_source_chunks_serial(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    node_id: NodeId,
    source_chunks: &[SourceChunk],
    reader_dynamic_filters: &[BoundedJoinLeafSourceStreamDynamicFilterStage],
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    max_attempts: usize,
    min_retry_chunk_row_groups: usize,
    grow_source_to_observed: bool,
) -> Result<Vec<(GpuDataFrame, ActiveReservation)>> {
    let batches = join_leaf_source_chunk_batches(
        source,
        source_chunks,
        reader_dynamic_filters,
        source_chunk_batch_target_bytes(context.memory_control_policy()),
    );
    let mut chunks = Vec::with_capacity(source_chunks.len());
    for batch in batches {
        if batch.is_singleton() {
            let chunk = batch.chunks.into_iter().next().expect("singleton source batch has one chunk");
            match read_join_leaf_source_chunk_with_retry(
                context,
                source,
                reservations,
                metrics,
                node_id,
                chunk,
                reader_dynamic_filters,
                1,
                max_attempts,
                min_retry_chunk_row_groups,
                grow_source_to_observed,
            ) {
                Ok(mut outputs) => chunks.append(&mut outputs),
                Err(error) => {
                    release_join_source_chunks_reverse(reservations, chunks, metrics);
                    return Err(error);
                },
            }
            continue;
        }
        match read_join_leaf_source_batch_or_split(
            context,
            source,
            reservations,
            metrics,
            node_id,
            batch,
            reader_dynamic_filters,
            max_attempts,
            min_retry_chunk_row_groups,
            grow_source_to_observed,
        ) {
            Ok(mut outputs) => chunks.append(&mut outputs),
            Err(error) => {
                release_join_source_chunks_reverse(reservations, chunks, metrics);
                return Err(error);
            },
        }
    }

    Ok(chunks)
}

fn read_join_leaf_source_batch_or_split(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    node_id: NodeId,
    batch: SourceChunkBatch,
    reader_dynamic_filters: &[BoundedJoinLeafSourceStreamDynamicFilterStage],
    max_attempts: usize,
    min_retry_chunk_row_groups: usize,
    grow_source_to_observed: bool,
) -> Result<Vec<(GpuDataFrame, ActiveReservation)>> {
    context.check_cancellation("nexus_query_engine.exec.bounded.join_leaf_source_chunk_batch_retry")?;
    if batch.is_singleton() {
        let chunk = batch.chunks.into_iter().next().expect("singleton source batch has one chunk");
        return read_join_leaf_source_chunk_with_retry(
            context,
            source,
            reservations,
            metrics,
            node_id,
            chunk,
            reader_dynamic_filters,
            1,
            max_attempts,
            min_retry_chunk_row_groups,
            grow_source_to_observed,
        );
    }
    match read_join_leaf_source_batch_attempt(
        context,
        source,
        reservations,
        metrics,
        node_id,
        &batch,
        reader_dynamic_filters,
        grow_source_to_observed,
    ) {
        Ok(output) => Ok(vec![output]),
        Err(error) if is_retryable_chunk_error(&error) => {
            metrics.record_source_chunk_retry();
            let Some((left, right)) = batch.split_for_retry() else {
                return Err(error);
            };
            let mut outputs = read_join_leaf_source_batch_or_split(
                context,
                source,
                reservations,
                metrics,
                node_id,
                left,
                reader_dynamic_filters,
                max_attempts,
                min_retry_chunk_row_groups,
                grow_source_to_observed,
            )?;
            match read_join_leaf_source_batch_or_split(
                context,
                source,
                reservations,
                metrics,
                node_id,
                right,
                reader_dynamic_filters,
                max_attempts,
                min_retry_chunk_row_groups,
                grow_source_to_observed,
            ) {
                Ok(mut right_outputs) => {
                    outputs.append(&mut right_outputs);
                    Ok(outputs)
                },
                Err(error) => {
                    release_join_source_chunks_reverse(reservations, outputs, metrics);
                    Err(error)
                },
            }
        },
        Err(error) => Err(error),
    }
}

fn source_chunk_batch_target_bytes(policy: MemoryControlPolicy) -> Option<u64> {
    match policy {
        MemoryControlPolicy::Bounded { max_source_chunk_bytes, .. } => Some(max_source_chunk_bytes),
        MemoryControlPolicy::Unbounded => None,
    }
}

fn capped_source_chunk_batch_target_bytes(max_batch_source_read_bytes: Option<u64>) -> Option<u64> {
    max_batch_source_read_bytes
        .map(|max_source_chunk_bytes| max_source_chunk_bytes.min(SOURCE_BATCH_READ_MAX_PLANNED_BYTES))
        .or(Some(SOURCE_BATCH_READ_MAX_PLANNED_BYTES))
}

fn source_chunk_batch_decode_scratch_risk(source: &LocalParquetSource, chunks: &[SourceChunk]) -> bool {
    source.reader_predicate.is_some()
        && chunks.iter().any(|chunk| {
            chunk.known_pure_filter_compressed_bytes.unwrap_or(0) > 0
                || chunk.known_pure_filter_uncompressed_bytes.unwrap_or(0) > 0
        })
}

fn source_chunk_batch_requested_bytes(policy: MemoryControlPolicy, batch: &SourceChunkBatch) -> u64 {
    batch.chunks.iter().fold(0u64, |total, chunk| total.saturating_add(chunk_requested_bytes(policy, chunk)))
}

fn read_join_leaf_source_batch_attempt(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    node_id: NodeId,
    batch: &SourceChunkBatch,
    reader_dynamic_filters: &[BoundedJoinLeafSourceStreamDynamicFilterStage],
    grow_source_to_observed: bool,
) -> Result<(GpuDataFrame, ActiveReservation)> {
    context.check_cancellation("nexus_query_engine.exec.bounded.join_leaf_source_chunk_batch")?;
    let first_chunk = batch.chunks.first().expect("source chunk batch is non-empty");
    for _chunk in &batch.chunks {
        metrics.record_source_chunk_attempt();
    }
    let requested_bytes = source_chunk_batch_requested_bytes(context.memory_control_policy(), batch);
    let mut reservation = reservations.try_reserve(
        join_source_read_reservation_request(
            context.memory_control_policy(),
            requested_bytes,
            reservations.budget(),
            node_id,
            first_chunk,
            1,
            1,
        )
        .with_requested_bytes(requested_bytes)
        .with_stage("join_source_read_batch"),
        metrics,
    )?;
    let source_token_budget = reservations.available_bytes_including(&reservation);
    let frame = match read_local_parquet_chunk_batch(
        context,
        source,
        batch,
        source_token_budget,
        "join_source_read",
        reader_dynamic_filters.len(),
        metrics,
    ) {
        Ok(frame) => frame,
        Err(error) => {
            reservations.release(reservation, metrics);
            return Err(error);
        },
    };
    if grow_source_to_observed {
        if let Err(error) = grow_join_source_reservation_to_observed_frame(
            &frame,
            &mut reservation,
            reservations,
            node_id,
            first_chunk,
            metrics,
            1,
            1,
        ) {
            drop(frame);
            reservations.release(reservation, metrics);
            return Err(error);
        }
    } else if let Err(error) = observe_device_memory_bytes(&frame, metrics) {
        drop(frame);
        reservations.release(reservation, metrics);
        return Err(error);
    }
    record_dynamic_filter_reader_predicate_batch_rows(reader_dynamic_filters, batch, &frame, metrics);
    metrics.record_source_chunk_commit(batch.row_group_count() as u64);
    Ok((frame, reservation))
}

fn read_join_leaf_source_chunk_with_retry(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    node_id: NodeId,
    chunk: SourceChunk,
    reader_dynamic_filters: &[BoundedJoinLeafSourceStreamDynamicFilterStage],
    attempt: usize,
    max_attempts: usize,
    min_retry_chunk_row_groups: usize,
    grow_source_to_observed: bool,
) -> Result<Vec<(GpuDataFrame, ActiveReservation)>> {
    match read_join_leaf_source_chunk_attempt(
        context,
        source,
        reservations,
        metrics,
        node_id,
        &chunk,
        reader_dynamic_filters,
        attempt,
        max_attempts,
        grow_source_to_observed,
    ) {
        Ok(output) => Ok(vec![output]),
        Err(error) if is_retryable_chunk_error(&error) => retry_join_leaf_source_chunk_after_error(
            context,
            source,
            reservations,
            metrics,
            node_id,
            chunk,
            reader_dynamic_filters,
            error,
            attempt,
            max_attempts,
            min_retry_chunk_row_groups,
            grow_source_to_observed,
        ),
        Err(error) => Err(error),
    }
}
fn retry_join_leaf_source_chunk_after_error(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    node_id: NodeId,
    chunk: SourceChunk,
    reader_dynamic_filters: &[BoundedJoinLeafSourceStreamDynamicFilterStage],
    error: QueryError,
    attempt: usize,
    max_attempts: usize,
    min_retry_chunk_row_groups: usize,
    grow_source_to_observed: bool,
) -> Result<Vec<(GpuDataFrame, ActiveReservation)>> {
    record_chunk_retry_error(&error, metrics);
    if attempt < max_attempts
        && let Some((left, right)) = split_retryable_chunk(&chunk, min_retry_chunk_row_groups)
    {
        metrics.record_source_chunk_retry();
        let mut outputs = read_join_leaf_source_chunk_with_retry(
            context,
            source,
            reservations,
            metrics,
            node_id,
            left,
            reader_dynamic_filters,
            attempt + 1,
            max_attempts,
            min_retry_chunk_row_groups,
            grow_source_to_observed,
        )?;
        outputs.extend(read_join_leaf_source_chunk_with_retry(
            context,
            source,
            reservations,
            metrics,
            node_id,
            right,
            reader_dynamic_filters,
            attempt + 1,
            max_attempts,
            min_retry_chunk_row_groups,
            grow_source_to_observed,
        )?);
        return Ok(outputs);
    }

    Err(chunk_retry_facts(&chunk, metrics).attach_to_error(error))
}
fn read_join_leaf_source_chunk_attempt(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    node_id: NodeId,
    chunk: &SourceChunk,
    reader_dynamic_filters: &[BoundedJoinLeafSourceStreamDynamicFilterStage],
    attempt: usize,
    max_attempts: usize,
    grow_source_to_observed: bool,
) -> Result<(GpuDataFrame, ActiveReservation)> {
    context.check_cancellation("nexus_query_engine.exec.bounded.join_leaf_source_chunk")?;
    metrics.record_source_chunk_attempt();
    if let Some(output) = try_read_cached_local_parquet_chunk(context, source, chunk, reservations, metrics)? {
        record_dynamic_filter_reader_predicate_rows(reader_dynamic_filters, chunk, &output.frame, metrics);
        metrics.record_source_chunk_commit(chunk.row_group_count() as u64);
        return Ok((output.frame, output.reservation));
    }
    let requested_bytes = chunk_requested_bytes(context.memory_control_policy(), chunk);
    let mut reservation = reservations.try_reserve(
        join_source_read_reservation_request(
            context.memory_control_policy(),
            requested_bytes,
            reservations.budget(),
            node_id,
            chunk,
            attempt,
            max_attempts,
        ),
        metrics,
    )?;
    let source_token_budget = reservations.available_bytes_including(&reservation);
    let frame = match read_local_parquet_chunk_with_reader_dynamic_filter_count(
        context,
        source,
        chunk,
        source_token_budget,
        "join_source_read",
        reader_dynamic_filters.len(),
        metrics,
    ) {
        Ok(frame) => frame,
        Err(error) => {
            match try_read_join_leaf_source_chunk_with_corrected_budget_after_error(
                context,
                source,
                chunk,
                node_id,
                &mut reservation,
                reservations,
                metrics,
                &error,
                reader_dynamic_filters,
                attempt,
                max_attempts,
            ) {
                Ok(Some(frame)) => frame,
                Ok(None) => {
                    reservations.release(reservation, metrics);
                    return Err(attach_join_source_read_resource_facts(error, node_id, chunk, attempt, max_attempts));
                },
                Err(retry_error) => {
                    reservations.release(reservation, metrics);
                    return Err(attach_join_source_read_resource_facts(
                        retry_error,
                        node_id,
                        chunk,
                        attempt.saturating_add(1),
                        max_attempts,
                    ));
                },
            }
        },
    };
    if grow_source_to_observed {
        if let Err(error) = grow_join_source_reservation_to_observed_frame(
            &frame,
            &mut reservation,
            reservations,
            node_id,
            chunk,
            metrics,
            attempt,
            max_attempts,
        ) {
            drop(frame);
            reservations.release(reservation, metrics);
            return Err(error);
        }
        let retained_device_bytes = reservation.bytes();
        if let Err(error) = insert_bounded_local_parquet_source_cache(
            context,
            source,
            chunk,
            &frame,
            &mut reservation,
            reservations,
            metrics,
            "join_source_read",
            attempt,
            max_attempts,
            retained_device_bytes,
        ) {
            drop(frame);
            reservations.release(reservation, metrics);
            return Err(error);
        }
    } else if let Err(error) = observe_device_memory_bytes(&frame, metrics) {
        drop(frame);
        reservations.release(reservation, metrics);
        return Err(error);
    }
    record_dynamic_filter_reader_predicate_rows(reader_dynamic_filters, chunk, &frame, metrics);
    metrics.record_source_chunk_commit(chunk.row_group_count() as u64);
    Ok((frame, reservation))
}

fn try_read_join_leaf_source_chunk_with_corrected_budget_after_error(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    chunk: &SourceChunk,
    node_id: NodeId,
    reservation: &mut ActiveReservation,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    error: &QueryError,
    reader_dynamic_filters: &[BoundedJoinLeafSourceStreamDynamicFilterStage],
    attempt: usize,
    max_attempts: usize,
) -> Result<Option<GpuDataFrame>> {
    context.check_cancellation("nexus_query_engine.exec.bounded.join_leaf_source_budget_retry")?;
    if attempt >= max_attempts {
        return Ok(None);
    }
    let Some(correction) = source_read_budget_correction_from_error(error, reservations, reservation) else {
        return Ok(None);
    };
    record_chunk_retry_error(error, metrics);
    metrics.record_source_chunk_retry();
    let request = join_source_read_reservation_request_for_boundary(
        context.memory_control_policy(),
        0,
        reservations.budget(),
        node_id,
        chunk,
        "join_source_read",
        attempt + 1,
        max_attempts,
    );
    try_grow_source_read_reservation_for_budget_correction(reservations, reservation, correction, request, metrics)?;
    metrics.record_source_chunk_attempt();
    read_local_parquet_chunk_with_reader_dynamic_filter_count(
        context,
        source,
        chunk,
        correction.corrected_budget_bytes(),
        "join_source_read",
        reader_dynamic_filters.len(),
        metrics,
    )
    .map(Some)
    .map_err(|error| correction.attach_to_error(error))
}

fn read_join_leaf_source_chunks_with_scheduler(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    node_id: NodeId,
    source_chunks: &[SourceChunk],
    reader_dynamic_filters: &[BoundedJoinLeafSourceStreamDynamicFilterStage],
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    max_attempts: usize,
    grow_source_to_observed: bool,
) -> Result<Vec<(GpuDataFrame, ActiveReservation)>> {
    let mut pending = VecDeque::new();
    let mut next_chunk_index = 0usize;
    context.check_cancellation("nexus_query_engine.exec.bounded.join_leaf_source_scheduler")?;
    start_next_required_join_leaf_read(
        context,
        source,
        source_chunks,
        node_id,
        reservations,
        metrics,
        &mut pending,
        &mut next_chunk_index,
        max_attempts,
    )?;

    let mut chunks = Vec::with_capacity(source_chunks.len());
    for (index, chunk) in source_chunks.iter().enumerate() {
        if let Err(error) = context.check_cancellation("nexus_query_engine.exec.bounded.join_leaf_source_scheduler") {
            abandon_join_leaf_reads(pending, reservations, metrics);
            release_join_source_chunks_reverse(reservations, chunks, metrics);
            return Err(error);
        }
        let scheduled =
            pending.pop_front().expect("join leaf source scheduler should have a read for the current chunk");
        let is_prefetch = scheduled.is_prefetch();
        let mut output = match scheduled.finish(reservations, metrics) {
            Ok(output) => output,
            Err(error) => {
                abandon_join_leaf_reads(pending, reservations, metrics);
                release_join_source_chunks_reverse(reservations, chunks, metrics);
                return Err(attach_join_source_read_resource_facts(error, node_id, chunk, 1, max_attempts));
            },
        };
        if is_prefetch {
            metrics.record_source_read_prefetch_completed();
        }
        if grow_source_to_observed {
            if let Err(error) = grow_join_source_reservation_to_observed_frame(
                &output.frame,
                &mut output.reservation,
                reservations,
                node_id,
                chunk,
                metrics,
                1,
                max_attempts,
            ) {
                drop(output.frame);
                reservations.release(output.reservation, metrics);
                abandon_join_leaf_reads(pending, reservations, metrics);
                release_join_source_chunks_reverse(reservations, chunks, metrics);
                return Err(error);
            }
            let retained_device_bytes = output.reservation.bytes();
            if let Err(error) = insert_bounded_local_parquet_source_cache(
                context,
                source,
                chunk,
                &output.frame,
                &mut output.reservation,
                reservations,
                metrics,
                "join_source_read",
                1,
                max_attempts,
                retained_device_bytes,
            ) {
                drop(output.frame);
                reservations.release(output.reservation, metrics);
                abandon_join_leaf_reads(pending, reservations, metrics);
                release_join_source_chunks_reverse(reservations, chunks, metrics);
                return Err(error);
            }
        } else if let Err(error) = observe_device_memory_bytes(&output.frame, metrics) {
            drop(output.frame);
            reservations.release(output.reservation, metrics);
            abandon_join_leaf_reads(pending, reservations, metrics);
            release_join_source_chunks_reverse(reservations, chunks, metrics);
            return Err(error);
        }
        record_dynamic_filter_reader_predicate_rows(reader_dynamic_filters, &output.chunk, &output.frame, metrics);
        metrics.record_source_chunk_commit(chunk.row_group_count() as u64);
        chunks.push((output.frame, output.reservation));

        if let Err(error) = fill_optional_join_leaf_reads(
            context,
            source,
            source_chunks,
            node_id,
            reservations,
            metrics,
            &mut pending,
            &mut next_chunk_index,
            max_attempts,
        ) {
            abandon_join_leaf_reads(pending, reservations, metrics);
            release_join_source_chunks_reverse(reservations, chunks, metrics);
            return Err(error);
        }
        if pending.is_empty()
            && index + 1 < source_chunks.len()
            && let Err(error) = start_next_required_join_leaf_read(
                context,
                source,
                source_chunks,
                node_id,
                reservations,
                metrics,
                &mut pending,
                &mut next_chunk_index,
                max_attempts,
            )
        {
            release_join_source_chunks_reverse(reservations, chunks, metrics);
            return Err(error);
        }
    }

    Ok(chunks)
}

fn stream_join_leaf_source_chunks_serial<F>(
    context: &ExecutionContext,
    stream: &BoundedJoinLeafLocalParquetSourceStream,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    consume: &mut F,
) -> Result<()>
where
    F: FnMut(ResidentGpuFrame, &mut ReservationTracker, &mut QueryMetrics) -> Result<()>,
{
    stream_join_leaf_source_chunks_serial_from(context, stream, &stream.chunks, reservations, metrics, consume)
}

fn stream_join_leaf_source_chunks_serial_from<F>(
    context: &ExecutionContext,
    stream: &BoundedJoinLeafLocalParquetSourceStream,
    chunks: &[SourceChunk],
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    consume: &mut F,
) -> Result<()>
where
    F: FnMut(ResidentGpuFrame, &mut ReservationTracker, &mut QueryMetrics) -> Result<()>,
{
    let batches = join_leaf_source_chunk_batches(
        &stream.source,
        chunks,
        &stream.reader_dynamic_filters,
        source_chunk_batch_target_bytes(context.memory_control_policy()),
    );
    for batch in batches {
        if batch.is_singleton() {
            let chunk = batch.chunks.into_iter().next().expect("singleton source batch has one chunk");
            stream_join_leaf_source_chunk_with_consumer_retry(
                context,
                stream,
                chunk,
                1,
                reservations,
                metrics,
                consume,
            )?;
            continue;
        }
        match stream_join_leaf_source_batch_or_split_with_consumer(
            context,
            stream,
            batch,
            reservations,
            metrics,
            consume,
        ) {
            Ok(()) => {},
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

fn stream_join_leaf_source_batch_or_split_with_consumer<F>(
    context: &ExecutionContext,
    stream: &BoundedJoinLeafLocalParquetSourceStream,
    batch: SourceChunkBatch,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    consume: &mut F,
) -> Result<()>
where
    F: FnMut(ResidentGpuFrame, &mut ReservationTracker, &mut QueryMetrics) -> Result<()>,
{
    context.check_cancellation("nexus_query_engine.exec.bounded.join_leaf_source_stream_batch_retry")?;
    if batch.is_singleton() {
        let chunk = batch.chunks.into_iter().next().expect("singleton source batch has one chunk");
        return stream_join_leaf_source_chunk_with_consumer_retry(
            context,
            stream,
            chunk,
            1,
            reservations,
            metrics,
            consume,
        );
    }
    match stream_join_leaf_source_batch_with_consumer(context, stream, &batch, reservations, metrics, consume) {
        Ok(()) => Ok(()),
        Err(error) if is_retryable_chunk_error(&error) => {
            metrics.record_source_chunk_retry();
            let Some((left, right)) = batch.split_for_retry() else {
                return Err(error);
            };
            stream_join_leaf_source_batch_or_split_with_consumer(
                context,
                stream,
                left,
                reservations,
                metrics,
                consume,
            )?;
            stream_join_leaf_source_batch_or_split_with_consumer(context, stream, right, reservations, metrics, consume)
        },
        Err(error) => Err(error),
    }
}

fn stream_join_leaf_source_batch_with_consumer<F>(
    context: &ExecutionContext,
    stream: &BoundedJoinLeafLocalParquetSourceStream,
    batch: &SourceChunkBatch,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    consume: &mut F,
) -> Result<()>
where
    F: FnMut(ResidentGpuFrame, &mut ReservationTracker, &mut QueryMetrics) -> Result<()>,
{
    let grow_source_to_observed = stream.stages.is_empty();
    let (frame, reservation) = read_join_leaf_source_batch_attempt(
        context,
        &stream.source,
        reservations,
        metrics,
        stream.node_id,
        batch,
        &stream.reader_dynamic_filters,
        grow_source_to_observed,
    )?;
    let input = ResidentGpuFrame { frame, reservation };
    let output = apply_join_leaf_source_stream_stages(context, stream, input, reservations, metrics)?;
    consume(output, reservations, metrics)
}

fn stream_join_leaf_source_chunk_with_consumer_retry<F>(
    context: &ExecutionContext,
    stream: &BoundedJoinLeafLocalParquetSourceStream,
    chunk: SourceChunk,
    attempt: usize,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    consume: &mut F,
) -> Result<()>
where
    F: FnMut(ResidentGpuFrame, &mut ReservationTracker, &mut QueryMetrics) -> Result<()>,
{
    context.check_cancellation("nexus_query_engine.exec.bounded.join_leaf_source_stream")?;
    let grow_source_to_observed = stream.stages.is_empty();
    let (frame, reservation) = match read_join_leaf_source_chunk_attempt(
        context,
        &stream.source,
        reservations,
        metrics,
        stream.node_id,
        &chunk,
        &stream.reader_dynamic_filters,
        attempt,
        stream.max_attempts,
        grow_source_to_observed,
    ) {
        Ok(output) => output,
        Err(error) if is_retryable_chunk_error(&error) => {
            return retry_join_leaf_source_stream_chunk_after_error(
                context,
                stream,
                chunk,
                error,
                attempt,
                reservations,
                metrics,
                consume,
            );
        },
        Err(error) => return Err(error),
    };
    let input = ResidentGpuFrame { frame, reservation };
    let output = match apply_join_leaf_source_stream_stages(context, stream, input, reservations, metrics) {
        Ok(output) => output,
        Err(error) if is_retryable_chunk_error(&error) => {
            return retry_join_leaf_source_stream_chunk_after_error(
                context,
                stream,
                chunk,
                error,
                attempt,
                reservations,
                metrics,
                consume,
            );
        },
        Err(error) => return Err(error),
    };
    match consume(output, reservations, metrics) {
        Ok(()) => Ok(()),
        Err(error) if is_retryable_chunk_error(&error) => retry_join_leaf_source_stream_chunk_after_error(
            context,
            stream,
            chunk,
            error,
            attempt,
            reservations,
            metrics,
            consume,
        ),
        Err(error) => Err(error),
    }
}

fn retry_join_leaf_source_stream_chunk_after_error<F>(
    context: &ExecutionContext,
    stream: &BoundedJoinLeafLocalParquetSourceStream,
    chunk: SourceChunk,
    error: QueryError,
    attempt: usize,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    consume: &mut F,
) -> Result<()>
where
    F: FnMut(ResidentGpuFrame, &mut ReservationTracker, &mut QueryMetrics) -> Result<()>,
{
    record_chunk_retry_error(&error, metrics);
    if attempt < stream.max_attempts
        && let Some((left, right)) = split_retryable_chunk(&chunk, stream.min_retry_chunk_row_groups)
    {
        metrics.record_source_chunk_retry();
        stream_join_leaf_source_chunk_with_consumer_retry(
            context,
            stream,
            left,
            attempt + 1,
            reservations,
            metrics,
            consume,
        )?;
        stream_join_leaf_source_chunk_with_consumer_retry(
            context,
            stream,
            right,
            attempt + 1,
            reservations,
            metrics,
            consume,
        )?;
        return Ok(());
    }

    Err(chunk_retry_facts(&chunk, metrics).attach_to_error(error))
}

fn stream_join_leaf_source_chunks_with_scheduler<F>(
    context: &ExecutionContext,
    stream: &BoundedJoinLeafLocalParquetSourceStream,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    consume: &mut F,
) -> Result<()>
where
    F: FnMut(ResidentGpuFrame, &mut ReservationTracker, &mut QueryMetrics) -> Result<()>,
{
    let grow_source_to_observed = stream.stages.is_empty();
    let mut pending = VecDeque::new();
    let mut next_chunk_index = 0usize;
    context.check_cancellation("nexus_query_engine.exec.bounded.join_leaf_source_stream_scheduler")?;
    start_next_required_join_leaf_read(
        context,
        &stream.source,
        &stream.chunks,
        stream.node_id,
        reservations,
        metrics,
        &mut pending,
        &mut next_chunk_index,
        stream.max_attempts,
    )?;

    for (index, chunk) in stream.chunks.iter().enumerate() {
        if let Err(error) =
            context.check_cancellation("nexus_query_engine.exec.bounded.join_leaf_source_stream_scheduler")
        {
            abandon_join_leaf_reads(pending, reservations, metrics);
            return Err(error);
        }
        let scheduled =
            pending.pop_front().expect("join leaf source scheduler should have a read for the current chunk");
        let is_prefetch = scheduled.is_prefetch();
        let mut output = match scheduled.finish(reservations, metrics) {
            Ok(output) => output,
            Err(error) if is_retryable_chunk_error(&error) => {
                if is_prefetch {
                    metrics.record_source_read_prefetch_abandoned();
                }
                abandon_join_leaf_reads(pending, reservations, metrics);
                return stream_join_leaf_source_chunks_serial_from(
                    context,
                    stream,
                    &stream.chunks[index..],
                    reservations,
                    metrics,
                    consume,
                );
            },
            Err(error) => {
                abandon_join_leaf_reads(pending, reservations, metrics);
                return Err(attach_join_source_read_resource_facts(
                    error,
                    stream.node_id,
                    chunk,
                    1,
                    stream.max_attempts,
                ));
            },
        };
        if is_prefetch {
            metrics.record_source_read_prefetch_completed();
        }
        if grow_source_to_observed {
            if let Err(error) = grow_join_source_reservation_to_observed_frame(
                &output.frame,
                &mut output.reservation,
                reservations,
                stream.node_id,
                chunk,
                metrics,
                1,
                stream.max_attempts,
            ) {
                drop(output.frame);
                reservations.release(output.reservation, metrics);
                abandon_join_leaf_reads(pending, reservations, metrics);
                return Err(error);
            }
            let retained_device_bytes = output.reservation.bytes();
            if let Err(error) = insert_bounded_local_parquet_source_cache(
                context,
                &stream.source,
                chunk,
                &output.frame,
                &mut output.reservation,
                reservations,
                metrics,
                "join_source_read",
                1,
                stream.max_attempts,
                retained_device_bytes,
            ) {
                drop(output.frame);
                reservations.release(output.reservation, metrics);
                abandon_join_leaf_reads(pending, reservations, metrics);
                return Err(error);
            }
        } else if let Err(error) = observe_device_memory_bytes(&output.frame, metrics) {
            drop(output.frame);
            reservations.release(output.reservation, metrics);
            abandon_join_leaf_reads(pending, reservations, metrics);
            return Err(error);
        }
        record_dynamic_filter_reader_predicate_rows(
            &stream.reader_dynamic_filters,
            &output.chunk,
            &output.frame,
            metrics,
        );
        metrics.record_source_chunk_commit(chunk.row_group_count() as u64);
        let input = ResidentGpuFrame { frame: output.frame, reservation: output.reservation };
        let output = match apply_join_leaf_source_stream_stages(context, stream, input, reservations, metrics) {
            Ok(output) => output,
            Err(error) if is_retryable_chunk_error(&error) => {
                abandon_join_leaf_reads(pending, reservations, metrics);
                return stream_join_leaf_source_chunks_serial_from(
                    context,
                    stream,
                    &stream.chunks[index..],
                    reservations,
                    metrics,
                    consume,
                );
            },
            Err(error) => {
                abandon_join_leaf_reads(pending, reservations, metrics);
                return Err(error);
            },
        };
        if let Err(error) = consume(output, reservations, metrics) {
            abandon_join_leaf_reads(pending, reservations, metrics);
            if is_retryable_chunk_error(&error) {
                return stream_join_leaf_source_chunks_serial_from(
                    context,
                    stream,
                    &stream.chunks[index..],
                    reservations,
                    metrics,
                    consume,
                );
            }
            return Err(error);
        }

        if let Err(error) = fill_optional_join_leaf_reads(
            context,
            &stream.source,
            &stream.chunks,
            stream.node_id,
            reservations,
            metrics,
            &mut pending,
            &mut next_chunk_index,
            stream.max_attempts,
        ) {
            abandon_join_leaf_reads(pending, reservations, metrics);
            return Err(error);
        }
        if pending.is_empty()
            && index + 1 < stream.chunks.len()
            && let Err(error) = start_next_required_join_leaf_read(
                context,
                &stream.source,
                &stream.chunks,
                stream.node_id,
                reservations,
                metrics,
                &mut pending,
                &mut next_chunk_index,
                stream.max_attempts,
            )
        {
            return Err(error);
        }
    }

    Ok(())
}
fn start_next_required_join_leaf_read(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    chunks: &[SourceChunk],
    node_id: NodeId,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    pending: &mut VecDeque<ScheduledLocalParquetRead>,
    next_chunk_index: &mut usize,
    max_attempts: usize,
) -> Result<()> {
    if let Some(chunk) = chunks.get(*next_chunk_index) {
        let requested_bytes = chunk_requested_bytes(context.memory_control_policy(), chunk);
        pending.push_back(start_join_leaf_scheduled_source_read(
            context,
            source,
            chunk,
            node_id,
            reservations,
            metrics,
            requested_bytes,
            "join_source_read",
            false,
            false,
            1,
            max_attempts,
        )?);
        *next_chunk_index += 1;
    }
    Ok(())
}
fn fill_optional_join_leaf_reads(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    chunks: &[SourceChunk],
    node_id: NodeId,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    pending: &mut VecDeque<ScheduledLocalParquetRead>,
    next_chunk_index: &mut usize,
    max_attempts: usize,
) -> Result<()> {
    // Keep only one unobserved optional source read ahead. The source decoded
    // output preflight is still an unresolved lower-level contract, so deeper
    // speculative reads can starve the next front-of-queue read when it grows
    // to observed bytes.
    let prefetch_target = usize::from(context.source_scheduler_policy().max_source_reads_in_flight() > 1);
    while pending.len() < prefetch_target {
        let Some(chunk) = chunks.get(*next_chunk_index) else {
            break;
        };
        let Some(requested_bytes) = join_leaf_prefetch_requested_bytes(context.memory_control_policy(), chunk) else {
            metrics.record_source_read_prefetch_skipped();
            break;
        };
        if !reservations.can_reserve(requested_bytes) {
            metrics.record_source_read_prefetch_skipped();
            break;
        }
        metrics.record_source_read_prefetch_started();
        pending.push_back(start_join_leaf_scheduled_source_read(
            context,
            source,
            chunk,
            node_id,
            reservations,
            metrics,
            requested_bytes,
            "join_source_read_prefetch",
            false,
            true,
            1,
            max_attempts,
        )?);
        *next_chunk_index += 1;
    }
    Ok(())
}
fn start_join_leaf_scheduled_source_read(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    chunk: &SourceChunk,
    node_id: NodeId,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    requested_bytes: u64,
    memory_boundary: &'static str,
    strict_token_budget: bool,
    prefetch: bool,
    attempt: usize,
    max_attempts: usize,
) -> Result<ScheduledLocalParquetRead> {
    context.check_cancellation("nexus_query_engine.exec.bounded.join_leaf_source_scheduler_start")?;
    metrics.record_source_chunk_attempt();
    if let Some(output) = try_read_cached_local_parquet_chunk(context, source, chunk, reservations, metrics)? {
        return Ok(ScheduledLocalParquetRead::ready(output, memory_boundary, prefetch));
    }
    let request = join_source_read_reservation_request_for_boundary(
        context.memory_control_policy(),
        requested_bytes,
        reservations.budget(),
        node_id,
        chunk,
        memory_boundary,
        attempt,
        max_attempts,
    );
    let reservation = reservations.try_reserve(request, metrics)?;
    let source_token_budget =
        if strict_token_budget { reservation.bytes() } else { reservations.available_bytes_including(&reservation) };
    ScheduledLocalParquetRead::spawn(
        context.clone(),
        source.clone(),
        chunk.clone(),
        reservation,
        source_token_budget,
        memory_boundary,
        prefetch,
        attempt,
        max_attempts,
    )
}

fn abandon_join_leaf_reads(
    mut pending: VecDeque<ScheduledLocalParquetRead>,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) {
    while let Some(prefetch) = pending.pop_back() {
        prefetch.abandon(reservations, metrics);
    }
}

fn attach_join_source_read_resource_facts(
    error: QueryError,
    node_id: NodeId,
    chunk: &SourceChunk,
    attempt: usize,
    max_attempts: usize,
) -> QueryError {
    error
        .with_fact_if_absent("memory_boundary", "join_source_read")
        .with_fact_if_absent("stage", "join_source_read")
        .with_fact_if_absent("node_id", node_id)
        .with_fact_if_absent("attempt", attempt)
        .with_fact_if_absent("max_attempts", max_attempts)
        .with_fact_if_absent("chunk_index", chunk.chunk_index)
        .with_fact_if_absent("file_index", chunk.file_index)
        .with_fact_if_absent("path", chunk.path.display())
        .with_fact_if_absent("row_groups", format_row_groups(chunk))
}

fn join_source_read_reservation_request(
    policy: MemoryControlPolicy,
    requested_bytes: u64,
    budget_bytes: u64,
    node_id: NodeId,
    chunk: &SourceChunk,
    attempt: usize,
    max_attempts: usize,
) -> MemoryReservationRequest {
    join_source_read_reservation_request_for_boundary(
        policy,
        requested_bytes,
        budget_bytes,
        node_id,
        chunk,
        "join_source_read",
        attempt,
        max_attempts,
    )
}

fn join_source_read_reservation_request_for_boundary(
    policy: MemoryControlPolicy,
    requested_bytes: u64,
    budget_bytes: u64,
    node_id: NodeId,
    chunk: &SourceChunk,
    memory_boundary: &'static str,
    attempt: usize,
    max_attempts: usize,
) -> MemoryReservationRequest {
    let mut request = MemoryReservationRequest::new(ReservationKind::SourceRead, requested_bytes, budget_bytes)
        .with_memory_boundary(memory_boundary)
        .with_node_id(node_id)
        .with_stage(memory_boundary)
        .with_chunk_index(chunk.chunk_index)
        .with_file_index(chunk.file_index)
        .with_path(chunk.path.display())
        .with_attempt(attempt, max_attempts);
    if !chunk.reads_all_row_groups {
        request = request.with_row_groups(chunk.row_group_indexes.clone());
    }
    let contract = source_read_reservation_contract(policy, chunk);
    request.with_source_read_contract(contract.quality.as_str(), contract.planning_bytes, contract.estimated_bytes)
}

fn observed_join_source_read_reservation_request(
    budget_bytes: u64,
    node_id: NodeId,
    chunk: &SourceChunk,
    attempt: usize,
    max_attempts: usize,
) -> MemoryReservationRequest {
    let mut request = MemoryReservationRequest::new(ReservationKind::SourceRead, 0, budget_bytes)
        .with_memory_boundary("join_source_read")
        .with_node_id(node_id)
        .with_stage("join_source_read")
        .with_chunk_index(chunk.chunk_index)
        .with_file_index(chunk.file_index)
        .with_path(chunk.path.display())
        .with_attempt(attempt, max_attempts);
    if !chunk.reads_all_row_groups {
        request = request.with_row_groups(chunk.row_group_indexes.clone());
    }
    request.with_source_read_contract(
        SourceReadReservationContractQuality::ObservedGrowShrink.as_str(),
        chunk.estimated_chunk_bytes,
        chunk.estimated_source_read_bytes,
    )
}
fn grow_join_source_reservation_to_observed_frame(
    frame: &GpuDataFrame,
    reservation: &mut ActiveReservation,
    reservations: &mut ReservationTracker,
    node_id: NodeId,
    chunk: &SourceChunk,
    metrics: &mut QueryMetrics,
    attempt: usize,
    max_attempts: usize,
) -> Result<()> {
    let observed_bytes = observe_device_memory_bytes(frame, metrics)?;
    if observed_bytes <= reservation.bytes() {
        reservations.shrink_to_observed_with_request(
            reservation,
            observed_bytes,
            observed_join_source_read_reservation_request(reservations.budget(), node_id, chunk, attempt, max_attempts),
            metrics,
        );
        return Ok(());
    }
    let delta = observed_bytes.saturating_sub(reservation.bytes());
    reservations.try_grow(
        reservation,
        observed_join_source_read_reservation_request(reservations.budget(), node_id, chunk, attempt, max_attempts)
            .with_requested_bytes(delta),
        metrics,
    )
}

fn apply_join_leaf_source_stream_stages(
    context: &ExecutionContext,
    stream: &BoundedJoinLeafLocalParquetSourceStream,
    mut input: ResidentGpuFrame,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    for stage in &stream.stages {
        input = match stage {
            BoundedJoinLeafSourceStreamStage::Filter { node_id, node, dynamic_filter } => {
                let rows_before = input.frame.num_rows() as u64;
                let output = execute_join_leaf_source_stream_unary(
                    context,
                    *node_id,
                    "join_source_filter",
                    "join_source_filter_output",
                    input,
                    reservations,
                    metrics,
                    |context, input, token, metrics| {
                        execute_filter_with_budget(
                            context,
                            node,
                            input,
                            token.expect("budget token is created for join source stream filter stages"),
                            metrics,
                        )
                    },
                )?;
                if let Some(dynamic_filter) = dynamic_filter {
                    metrics.record_dynamic_filter_source_post_decode_stage(
                        &dynamic_filter.route,
                        &dynamic_filter.filter,
                        Some(rows_before),
                        Some(output.frame.num_rows() as u64),
                    );
                }
                Ok(output)
            },
            BoundedJoinLeafSourceStreamStage::Projection { node_id, node } => execute_join_leaf_source_stream_unary(
                context,
                *node_id,
                "join_source_projection",
                "join_source_projection_output",
                input,
                reservations,
                metrics,
                |context, input, token, metrics| {
                    execute_projection_with_budget(
                        context,
                        node,
                        input,
                        token.expect("budget token is created for join source stream projection stages"),
                        metrics,
                    )
                },
            ),
        }?;
    }
    Ok(input)
}
fn execute_join_leaf_source_stream_unary(
    context: &ExecutionContext,
    node_id: NodeId,
    stage: &'static str,
    output_boundary: &'static str,
    input: ResidentGpuFrame,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    execute: impl FnOnce(
        &ExecutionContext,
        GpuDataFrame,
        Option<&ExecutionBudgetToken>,
        &mut QueryMetrics,
    ) -> Result<GpuDataFrame>,
) -> Result<ResidentGpuFrame> {
    let input_reserved_bytes = input.reservation.bytes();
    let mut work_reservation = match reservations.try_reserve_work(
        MemoryReservationRequest::new(ReservationKind::OperatorOutput, input_reserved_bytes, reservations.budget())
            .with_memory_boundary(stage)
            .with_node_id(node_id)
            .with_stage(stage)
            .with_attempt(1, 1),
        metrics,
    ) {
        Ok(reservation) => reservation,
        Err(error) => {
            drop(input.frame);
            reservations.release(input.reservation, metrics);
            return Err(error);
        },
    };
    let token = match execution_budget_token_for_boundary(
        context,
        reservations.available_bytes_including(&work_reservation),
        stage,
        "nexus_query_engine.bounded.join.source_stream_unary.budget_token",
        "failed to create bounded native join source stream operator budget token",
    ) {
        Ok(token) => token,
        Err(error) => {
            reservations.release(work_reservation, metrics);
            reservations.release(input.reservation, metrics);
            return Err(error.with_fact_if_absent("node_id", node_id));
        },
    };
    let result = execute(context, input.frame, Some(&token), metrics);
    let frame = match result {
        Ok(frame) => {
            reservations.release(input.reservation, metrics);
            frame
        },
        Err(error) => {
            reservations.release(work_reservation, metrics);
            reservations.release(input.reservation, metrics);
            return Err(error.with_fact_if_absent("node_id", node_id));
        },
    };
    let output_bytes = match observe_device_memory_bytes(&frame, metrics) {
        Ok(bytes) => bytes,
        Err(error) => {
            drop(frame);
            reservations.release(work_reservation, metrics);
            return Err(error.with_fact_if_absent("node_id", node_id));
        },
    };
    let output_request =
        MemoryReservationRequest::new(ReservationKind::OperatorOutput, output_bytes, reservations.budget())
            .with_memory_boundary(output_boundary)
            .with_node_id(node_id)
            .with_stage(stage)
            .with_attempt(1, 1);
    if output_bytes > work_reservation.bytes() {
        let delta = output_bytes.saturating_sub(work_reservation.bytes());
        if let Err(error) =
            reservations.try_grow(&mut work_reservation, output_request.clone().with_requested_bytes(delta), metrics)
        {
            drop(frame);
            reservations.release(work_reservation, metrics);
            return Err(error);
        }
    } else {
        reservations.shrink_to_observed_with_request(
            &mut work_reservation,
            output_bytes,
            output_request.clone(),
            metrics,
        );
    }
    let output_reservation = reservations.transfer(work_reservation, output_request, metrics);
    Ok(ResidentGpuFrame { frame, reservation: output_reservation })
}

fn concatenate_join_source_chunks(
    context: &ExecutionContext,
    node_id: NodeId,
    mut chunks: Vec<(GpuDataFrame, ActiveReservation)>,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    if chunks.len() == 1 {
        let (frame, reservation) = chunks.pop().expect("single chunk checked");
        return Ok(ResidentGpuFrame { frame, reservation });
    }

    let output_reservation_bytes =
        chunks.iter().fold(0u64, |total, (_frame, reservation)| total.saturating_add(reservation.bytes()));
    let mut output_reservation = match reservations.try_reserve(
        MemoryReservationRequest::new(ReservationKind::SourceRead, output_reservation_bytes, reservations.budget())
            .with_memory_boundary("join_source_concat_output")
            .with_node_id(node_id)
            .with_stage("join_source_concat_output")
            .with_attempt(1, 1),
        metrics,
    ) {
        Ok(reservation) => reservation,
        Err(error) => {
            release_join_source_chunks_reverse(reservations, chunks, metrics);
            return Err(error);
        },
    };

    let mut input_reservations = Vec::with_capacity(chunks.len());
    let mut frames = Vec::with_capacity(chunks.len());
    for (frame, reservation) in chunks {
        frames.push(frame.into_frame());
        input_reservations.push(reservation);
    }
    let chunk_count = input_reservations.len();
    let first_frame = frames.remove(0);
    let rest_frame_refs = frames.iter().collect::<Vec<_>>();
    let overrides = execution_overrides(context);
    let timer = Instant::now();
    let output = match first_frame.with_execution(&overrides).concat_many(&rest_frame_refs) {
        Ok(output) => output,
        Err(error) => {
            drop(rest_frame_refs);
            drop(frames);
            drop(first_frame);
            release_join_source_reservations_reverse(reservations, input_reservations, metrics);
            reservations.release(output_reservation, metrics);
            return Err(QueryError::from_cudf(
                "nexus_query_engine.bounded.join.source_concat",
                "failed to concatenate bounded join leaf local parquet source chunks",
                error,
            )
            .with_fact("node_id", node_id)
            .with_fact("chunk_count", chunk_count));
        },
    };
    metrics.operator_time += timer.elapsed();

    let output_frame = GpuDataFrame::new(output);
    let output_bytes = match observe_device_memory_bytes(&output_frame, metrics) {
        Ok(bytes) => bytes,
        Err(error) => {
            drop(output_frame);
            drop(rest_frame_refs);
            drop(frames);
            drop(first_frame);
            release_join_source_reservations_reverse(reservations, input_reservations, metrics);
            reservations.release(output_reservation, metrics);
            return Err(error);
        },
    };
    if output_bytes > output_reservation.bytes() {
        let delta = output_bytes.saturating_sub(output_reservation.bytes());
        if let Err(error) = reservations.try_grow(
            &mut output_reservation,
            MemoryReservationRequest::new(ReservationKind::SourceRead, delta, reservations.budget())
                .with_memory_boundary("join_source_concat_output")
                .with_node_id(node_id)
                .with_stage("join_source_concat_output")
                .with_attempt(1, 1),
            metrics,
        ) {
            drop(output_frame);
            drop(rest_frame_refs);
            drop(frames);
            drop(first_frame);
            release_join_source_reservations_reverse(reservations, input_reservations, metrics);
            reservations.release(output_reservation, metrics);
            return Err(error);
        }
    }
    drop(rest_frame_refs);
    drop(frames);
    drop(first_frame);
    release_join_source_reservations_reverse(reservations, input_reservations, metrics);
    Ok(ResidentGpuFrame { frame: output_frame, reservation: output_reservation })
}

fn release_join_source_chunks_reverse(
    reservations: &mut ReservationTracker,
    mut chunks: Vec<(GpuDataFrame, ActiveReservation)>,
    metrics: &mut QueryMetrics,
) {
    while let Some((frame, reservation)) = chunks.pop() {
        drop(frame);
        reservations.release(reservation, metrics);
    }
}

fn release_join_source_reservations_reverse(
    reservations: &mut ReservationTracker,
    mut active: Vec<ActiveReservation>,
    metrics: &mut QueryMetrics,
) {
    while let Some(reservation) = active.pop() {
        reservations.release(reservation, metrics);
    }
}
