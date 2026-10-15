//! Deferred local-Parquet leaves under compiled join and aggregate tasks.
//!
//! A join leaf is a Parquet source whose chunks are read while the join runs.
//! Its chunks are not materialized before the join task starts. This module owns
//! the stream description (`BoundedJoinLeafLocalParquetSourceStream`), the
//! stages that stay after decoding because the reader cannot evaluate them,
//! the dynamic-filter rewrite that narrows row groups from a completed build
//! side, and the choice between the three read drivers in `stream`,
//! `read_serial` and `scheduler`.
//!
//! Serial reads decode on the coordinator; scheduled reads run as source-read
//! jobs under the attempt's auxiliary permits and the coordinator blocks in
//! `finish` for each one. A chunk's source commit is published only after its
//! consumer accepted the frame on the serial path; the scheduler path commits
//! at decode time and the retry contract in `stream` says what each consumer
//! must discard on error.

use std::collections::{BTreeMap, BTreeSet};

use arrow::datatypes::SchemaRef as ArrowSchemaRef;
use cudf::Series;
use query_planning::{
    dynamic_filter::{
        DynamicFilter, DynamicFilterConsumerLowering, DynamicFilterExecutionOptions, DynamicFilterId,
        DynamicFilterRoute, DynamicFilterSet,
        apply_dynamic_filters_to_local_parquet_source_with_reader_predicate_filter,
        dynamic_filter_predicate_for_schema, dynamic_filter_reader_predicate_should_merge,
    },
    error::QueryErrorCode,
    plan::AggregateNode,
    source::{ParquetReadStrategy, SourceFooterFacts},
};
use query_runtime::metrics::DynamicFilterSourceSnapshot;

use super::{
    BoundedExecutionAttempt, BoundedSourceBatchingPolicy, ExecutionContext, ExecutionMetrics, FilterNode, GpuDataFrame,
    Instant, LocalParquetSource, MemoryControlPolicy, NodeId, ProjectionNode, QueryError, ResidentGpuFrame, Result,
    SourceChunk, SourceChunkBatch, SourceChunkPlanningPolicy, SourceTaskMemoryModel,
    annotate_source_chunk_batching_reasons, apply_source_adjacent_aggregate_chunk_caps, chunk_retry_facts,
    cudf_query_error, execute_projection_with_budget, is_retryable_chunk_error, join_leaf_prefetch_admitted,
    observe_device_memory_bytes, plan_local_parquet_source_chunks_with_execution_context, plan_source_chunk_batches,
    record_bounded_parquet_source_metrics, record_chunk_retry_error, split_retryable_chunk,
    validate_local_parquet_source,
};
use crate::{
    chunk::{plan_local_parquet_source_chunks_with_metadata_cache_and_context, source_footer_metadata},
    exec::{
        bounded::BoundedPlan,
        operators::PreparedDynamicFilter,
        source::{
            ScheduledLocalParquetRead, attach_row_group_facts_if_absent, parquet_chunk_path_fact,
            read_local_parquet_chunk_batch, read_local_parquet_chunk_with_reader_dynamic_filter_count,
            read_local_parquet_source,
        },
    },
    rapids_backend::rapids_analysis_context,
    source_chunk::SourceMetadataCache,
};

mod batching;
mod concat;
mod dynamic_filter;
mod planning;
mod read_serial;
mod scheduler;
mod stages;
mod stream;

use batching::{
    annotate_join_leaf_source_chunk_batching_reasons, join_leaf_source_chunk_batches_for_execution,
    materialized_source_chunk_batches_for_execution, source_chunk_batch_target_bytes, source_chunk_batches,
    stream_has_multi_source_batches,
};
use concat::concatenate_join_source_chunks;
use dynamic_filter::{
    deferred_large_small_in_list_reader_predicate_routes, merged_reader_predicate_route_ids,
    record_dynamic_filter_reader_predicate_batch_rows, record_dynamic_filter_reader_predicate_rows,
    source_post_decode_dynamic_filter_stages, source_reader_predicate_dynamic_filter_stages,
};
pub(in crate::exec::bounded) use planning::{
    plan_bounded_aggregate_input_local_parquet_source_stream, plan_bounded_join_leaf_local_parquet_source_stream,
    plan_bounded_shared_join_aggregate_local_parquet_source_stream,
    plan_bounded_unary_chain_local_parquet_source_stream,
};
use read_serial::{
    read_join_leaf_source_batch_attempt, read_join_leaf_source_batches_serial, read_join_leaf_source_chunk_attempt,
    read_join_leaf_source_chunks_serial,
};
use scheduler::{
    abandon_join_leaf_reads, fill_optional_join_leaf_reads, read_join_leaf_source_chunks_with_scheduler,
    start_join_leaf_prefetch_source_read, start_next_required_join_leaf_read,
};
use stages::apply_join_leaf_source_stream_stages;
pub(in crate::exec::bounded) use stream::bounded_join_leaf_local_parquet_source_frame_stream;
use stream::{
    execute_uncommitted_source_chunk, stream_join_leaf_source_chunk_with_consumer_retry,
    stream_join_leaf_source_chunks_serial, stream_join_leaf_source_chunks_with_scheduler,
};

/// Names the join source read as the failing boundary so a resource error
/// from a leaf read is attributed to the leaf, with the chunk and attempt that
/// failed. The join operator above the leaf is not the attributed boundary.
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
        .with_fact_if_absent("path", parquet_chunk_path_fact(chunk));
    attach_row_group_facts_if_absent(error, chunk)
}

/// Deferred Parquet leaf work that compiled join tasks can materialize in chunks.
///
/// The value retains device keys for post-decode dynamic filters and is cloned
/// into each compiled task that reads it. `chunks` is the planned chunk
/// list after any dynamic-filter replan, `stages` run in order on each decoded
/// chunk, and `reader_dynamic_filters` are the routes already merged into the
/// source's reader predicate, kept for the metrics that compare reader and
/// post-decode filtering. `max_attempts` and `min_retry_chunk_row_groups` are
/// taken from the memory control policy at planning time so retry shape does
/// not depend on which task reads the leaf.
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

/// A source read started before its consumer task runs, so a remote fetch
/// overlaps the build task. The owner must call `finish` or `abandon`; both
/// join the read, and `abandon` also discards the decoded frame under the
/// execution attempt so its device bytes are observed as released.
pub(in crate::exec::bounded) struct BoundedJoinLeafLocalParquetSourcePrefetch {
    scheduled: ScheduledLocalParquetRead,
}

/// Post-decode work that stays with the leaf stream because it cannot all be
/// pushed into the Parquet reader. A `Filter` carries its dynamic filter when
/// the route was not merged into the reader predicate, so the stage can
/// report post-decode selectivity for that route.
#[derive(Clone)]
enum BoundedJoinLeafSourceStreamStage {
    Filter {
        node_id: NodeId,
        node: PreparedDynamicFilter,
        dynamic_filter: Option<Box<BoundedJoinLeafSourceStreamDynamicFilterStage>>,
    },
    Projection {
        node_id: NodeId,
        node: ProjectionNode,
    },
}

/// Route and filter pair retained for reader-predicate and post-decode
/// dynamic-filter metrics.
#[derive(Clone)]
struct BoundedJoinLeafSourceStreamDynamicFilterStage {
    route: DynamicFilterRoute,
    filter: DynamicFilter,
}

impl BoundedJoinLeafLocalParquetSourceStream {
    /// Borrows the source-adjacent aggregate contract from this already
    /// planned occurrence. Arbitrary unary chains and dynamic-filter routes
    /// retain their stream interpreter when that contract cannot represent them.
    pub(in crate::exec::bounded) fn aggregate_plan<'a>(
        &'a self,
        aggregate_id: NodeId,
        aggregate: &'a AggregateNode,
        input_schema: &'a ArrowSchemaRef,
        output_schema: &'a ArrowSchemaRef,
    ) -> Option<BoundedPlan<'a>> {
        if !self.reader_dynamic_filters.is_empty() {
            return None;
        }
        let mut pre_filter_projections = Vec::new();
        let mut post_filter_projections = Vec::new();
        let mut filter = None;
        for stage in &self.stages {
            match stage {
                BoundedJoinLeafSourceStreamStage::Filter {
                    node: PreparedDynamicFilter::Expression(node),
                    dynamic_filter: None,
                    ..
                } if filter.is_none() => {
                    filter = Some(node);
                },
                BoundedJoinLeafSourceStreamStage::Filter { .. } => return None,
                BoundedJoinLeafSourceStreamStage::Projection { node, .. } => {
                    let stages =
                        if filter.is_some() { &mut post_filter_projections } else { &mut pre_filter_projections };
                    stages.push((node, node.output_schema()));
                },
            }
        }
        Some(BoundedPlan {
            source_id: self.node_id,
            source: &self.source,
            pre_filter_projections,
            filter,
            post_filter_projections,
            aggregate_id,
            aggregate_input_schema: input_schema,
            aggregate: (aggregate, output_schema),
            post_aggregate_stages: Vec::new(),
            sort: None,
            limit: None,
        })
    }

    pub(in crate::exec::bounded) fn chunks(&self) -> &[SourceChunk] {
        &self.chunks
    }

    /// Executes one logical chunk, including source-owned split retries. The
    /// consumer must keep outputs private until this call succeeds.
    pub(in crate::exec::bounded) fn execute_chunk<F>(
        &self,
        execution: &mut BoundedExecutionAttempt<'_>,
        chunk: SourceChunk,
        consume: &mut F,
    ) -> Result<u64>
    where
        F: FnMut(&SourceChunk, usize, ResidentGpuFrame, &mut BoundedExecutionAttempt<'_>) -> Result<()>,
    {
        execute_uncommitted_source_chunk(execution, self, chunk, 1, consume)
    }

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

    pub(in crate::exec::bounded) fn estimated_source_rows(&self) -> Option<u64> {
        self.chunks.iter().try_fold(0_u64, |total, chunk| total.checked_add(chunk.known_row_count?))
    }

    fn output_column_count(&self) -> usize {
        self.stages
            .iter()
            .rev()
            .find_map(|stage| match stage {
                BoundedJoinLeafSourceStreamStage::Projection { node, .. } => Some(node.output_schema().fields().len()),
                BoundedJoinLeafSourceStreamStage::Filter { .. } => None,
            })
            .unwrap_or_else(|| self.source.schema.fields().len())
    }

    pub(in crate::exec::bounded) fn dynamic_filter_source_snapshot(&self) -> DynamicFilterSourceSnapshot {
        DynamicFilterSourceSnapshot {
            row_groups: self.chunks.iter().map(|chunk| chunk.row_group_count() as u64).sum(),
            source_bytes: self.estimated_source_read_bytes(),
        }
    }

    pub(in crate::exec::bounded) fn has_stages(&self) -> bool {
        !self.stages.is_empty()
    }

    pub(in crate::exec::bounded) fn with_filter_stage(mut self, node_id: NodeId, node: FilterNode) -> Self {
        self.stages.push(BoundedJoinLeafSourceStreamStage::Filter {
            node_id,
            node: PreparedDynamicFilter::Expression(node),
            dynamic_filter: None,
        });
        self
    }

    /// Appends the post-decode filter of a dynamic-filter route whose
    /// consumer is the filter plan node `node_id`.
    pub(in crate::exec::bounded) fn with_filter_dynamic_filter_stage(
        mut self,
        node_id: NodeId,
        node: PreparedDynamicFilter,
        route: DynamicFilterRoute,
        filter: DynamicFilter,
    ) -> Self {
        let dynamic_filter = Some(Box::new(BoundedJoinLeafSourceStreamDynamicFilterStage { route, filter }));
        self.stages.push(BoundedJoinLeafSourceStreamStage::Filter { node_id, node, dynamic_filter });
        self
    }

    pub(in crate::exec::bounded) fn with_projection_stage(mut self, node_id: NodeId, node: ProjectionNode) -> Self {
        self.stages.push(BoundedJoinLeafSourceStreamStage::Projection { node_id, node });
        self
    }

    /// Routes whose `IN` list is too large to become a reader predicate and
    /// is therefore applied after decoding. Decided before the replan so the
    /// same set drives both the source rewrite and the stage list.
    pub(in crate::exec::bounded) fn deferred_reader_predicate_routes(
        &self,
        routes: &[DynamicFilterRoute],
        filter_set: &DynamicFilterSet,
        options: DynamicFilterExecutionOptions,
    ) -> BTreeSet<DynamicFilterId> {
        let routes = routes.iter().collect::<Vec<_>>();
        deferred_large_small_in_list_reader_predicate_routes(&self.source, &routes, filter_set, options)
    }

    /// Applies dynamic filters produced by a completed build side to this leaf.
    ///
    /// Routes that can be reader predicates are merged into the source and
    /// prune row groups during the replan; the remaining routes become filter
    /// stages prepended to the existing ones, so they run on the decoded chunk
    /// before the leaf's own filter. The before and after chunk snapshots feed
    /// the dynamic-filter source metrics. The leaf's file list never changes,
    /// only each file's row-group selection.
    pub(in crate::exec::bounded) fn with_dynamic_filters_and_deferred_reader_predicates(
        mut self,
        context: &ExecutionContext,
        routes: &[&DynamicFilterRoute],
        filter_set: &DynamicFilterSet,
        keysets: &BTreeMap<DynamicFilterId, Series>,
        deferred_reader_predicate_routes: &BTreeSet<DynamicFilterId>,
        options: DynamicFilterExecutionOptions,
    ) -> Result<Self> {
        if routes.is_empty() {
            return Ok(self);
        }
        // One cache, one footer load for `self.source`: the same per-file
        // footer slots serve the merge check, the source rewrite, and the
        // re-plan below, because dynamic filtering only narrows each file's
        // `row_groups` selection and never changes `source.files` itself.
        let mut metadata_cache = SourceMetadataCache::default();
        let footer_metadata = source_footer_metadata(&self.source, &mut metadata_cache, Some(context))?;
        let footer_refs = footer_metadata.iter().map(|metadata| metadata.as_deref()).collect::<Vec<_>>();
        let footers = SourceFooterFacts::new(&footer_refs);

        let merged_reader_predicate_routes = merged_reader_predicate_route_ids(
            &self.source,
            routes,
            filter_set,
            deferred_reader_predicate_routes,
            options,
            footers,
        )?;
        let before = self.dynamic_filter_source_snapshot();
        let source = apply_dynamic_filters_to_local_parquet_source_with_reader_predicate_filter(
            &self.source,
            routes,
            filter_set,
            |route, _filter| merged_reader_predicate_routes.contains(&route.id),
            rapids_analysis_context(context.planning_policy()),
            footers,
        )?;
        let mut chunk_plan = plan_local_parquet_source_chunks_with_metadata_cache_and_context(
            self.node_id,
            &source,
            self.planning_policy,
            &mut metadata_cache,
            Some(context),
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
                .map(|(route, filter, node)| {
                    let column = route.consumer_columns.first().expect("post-decode route has a consumer column");
                    let prepared = PreparedDynamicFilter::bind(
                        self.source.schema.as_ref(),
                        column,
                        node.input,
                        filter,
                        keysets.get(&route.id),
                    )?
                    .ok_or_else(|| {
                        QueryError::new(
                            QueryErrorCode::InvalidPlan,
                            "query_engine.dynamic_filter.consumer_keyset",
                            "validated source dynamic filter has an incompatible resident key domain",
                        )
                    })?;
                    Ok(BoundedJoinLeafSourceStreamStage::Filter {
                        node_id: route.consumer.node_id(),
                        node: prepared,
                        dynamic_filter: Some(Box::new(BoundedJoinLeafSourceStreamDynamicFilterStage {
                            route: route.clone(),
                            filter: filter.clone(),
                        })),
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            stages.extend(self.stages);
            self.stages = stages;
        }
        Ok(self)
    }
}

impl BoundedJoinLeafLocalParquetSourcePrefetch {
    /// Joins the read and commits its chunk. Blocks until the worker returns
    /// and the frame's readiness event has fired on the device.
    pub(in crate::exec::bounded) fn finish(
        self,
        execution: &mut BoundedExecutionAttempt<'_>,
    ) -> Result<ResidentGpuFrame> {
        let is_prefetch = self.scheduled.is_prefetch();
        let output = self.scheduled.finish()?;
        if is_prefetch {
            execution.metrics().record_source_read_prefetch_completed();
        }
        let device_bytes = observe_device_memory_bytes(&output.frame, execution.metrics())?;
        execution.metrics().record_source_chunk_commit(
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

/// Starts the leaf's read ahead of its consumer task when that is worthwhile.
///
/// Only a single-chunk leaf read entirely with remote-direct reads and with
/// no post-decode stages or reader dynamic filters qualifies: the remote fetch
/// is the latency worth overlapping, one chunk needs no ordering with later
/// reads, and dynamic filters would be lost on a read that already started.
/// `Ok(None)` means no read was started.
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
    Ok(start_join_leaf_prefetch_source_read(execution, &stream.source, chunk)?
        .map(|scheduled| BoundedJoinLeafLocalParquetSourcePrefetch { scheduled }))
}

fn join_leaf_source_prefetch_remote_reader_admitted(source: &LocalParquetSource) -> bool {
    !source.files.is_empty()
        && source
            .files
            .iter()
            .all(|file| matches!(file.resolved_source.read_strategy(), ParquetReadStrategy::RemoteDirect))
}

/// Reads the whole leaf into one resident frame, for consumers that need both
/// join inputs at once. Chunks are read through the serial or scheduled driver
/// and concatenated once at the end, so the peak holds every decoded chunk
/// plus the concatenation.
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
        stream_result?;
        return concatenate_join_source_chunks(execution, node_id, chunks);
    }

    let chunks = read_join_leaf_source_stream_chunks(execution, &stream)?;
    concatenate_join_source_chunks(execution, node_id, chunks)
}

/// Pushes the leaf's chunks to `consume` in chunk order, choosing the driver.
///
/// Multi-chunk batches read serially because the scheduler issues one chunk
/// per job; otherwise a multi-chunk leaf with prefetch enabled uses the
/// scheduler, which keeps one read ahead of the consumer. Frames `consume`
/// already accepted stay with the consumer on error.
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

/// Pushes chunks one at a time with the single-chunk split retry and no
/// batching, for consumers whose retry contract is per chunk.
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

/// Reads every chunk and returns the decoded frames with their device bytes.
///
/// The scheduler never splits a chunk, so a retryable chunk error on that
/// path restarts the whole read serially, where the split retry applies. The
/// stages run after all chunks are resident; the input frame of each stage is
/// dropped as its output replaces it.
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
            let (frame, input_device_bytes) = pending.remove(0);
            let output = apply_join_leaf_source_stream_stages(
                execution,
                stream,
                ResidentGpuFrame::with_device_bytes(frame, input_device_bytes),
            );
            match output {
                Ok(output) => {
                    let device_bytes = output.device_bytes();
                    transformed.push((output.frame, device_bytes));
                },
                Err(error) => {
                    return Err(error);
                },
            }
        }
        chunks = transformed;
    }
    Ok(chunks)
}

/// The leaf's zero-row frame with the source's execution schema. Reading the
/// source whose files all have an empty row-group selection yields that frame
/// through the ordinary source path, so column coercion and missing-column
/// materialization are not duplicated here.
fn empty_join_leaf_source_frame(
    execution: &mut BoundedExecutionAttempt<'_>,
    stream: &BoundedJoinLeafLocalParquetSourceStream,
) -> Result<ResidentGpuFrame> {
    let frame = read_local_parquet_source(execution.context(), stream.node_id, &stream.source, execution.metrics())?;
    let device_bytes = observe_device_memory_bytes(&frame, execution.metrics())?;
    Ok(ResidentGpuFrame::with_device_bytes(frame, device_bytes))
}
