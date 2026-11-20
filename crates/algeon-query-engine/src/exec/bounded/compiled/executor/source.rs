use std::collections::BTreeSet;

use query_planning::{
    error::Result,
    pipeline::{
        PipelineEdgeKind, PipelineTask, PipelineTaskId, PipelineTaskKind, ResidentResourceId, ResourceSharing,
        SourceOutput, TaskPhysical,
    },
    plan::{AggregateNode, NodeId, PlanNode},
    source::{LocalParquetSource, SourceNode},
};

use super::{CompiledGpuResource, CompiledGpuTaskExecutor};
use crate::exec::bounded::{
    aggregate_source_stream_reads_ahead,
    compiled::{
        helpers::{attach_task_facts, invalid, plan_node, single_output_resource},
        lanes::AggregateLaneDispatch,
    },
    execute_bounded_gpu_resident_source, execute_bounded_literal_batch_source, execute_bounded_local_parquet_source,
    materialize_bounded_join_leaf_local_parquet_source_stream,
    plan_bounded_aggregate_input_local_parquet_source_stream, plan_bounded_join_leaf_local_parquet_source_stream,
    plan_bounded_shared_join_aggregate_local_parquet_source_stream,
    plan_bounded_unary_chain_local_parquet_source_stream, source_read_ahead_applies,
    try_start_bounded_join_leaf_local_parquet_source_prefetch,
};

impl CompiledGpuTaskExecutor<'_> {
    pub(super) fn execute_source_task(
        &mut self,
        task: &PipelineTask,
    ) -> Result<Vec<(ResidentResourceId, CompiledGpuResource)>> {
        let node_id = plan_node(task)?;
        let output = single_output_resource(task)?;
        let node = self.plan.node(node_id).ok_or_else(|| {
            invalid("compiled source task references a missing plan node")
                .with_fact("compiled_task_id", task.id())
                .with_fact("plan_node_id", node_id)
        })?;
        let resource = match &node.kind {
            PlanNode::Source(source) => match source.as_ref() {
                SourceNode::LocalParquet(source) => self.execute_local_parquet_source(task, node_id, output, source),
                SourceNode::GpuResident(source) => {
                    execute_bounded_gpu_resident_source(&mut self.execution, source, &self.gpu_inputs).and_then(
                        |frame| {
                            self.retire_gpu_input_read(&source.input_name)?;
                            Ok(CompiledGpuResource::new(frame))
                        },
                    )
                },
                SourceNode::LiteralBatch(source) => {
                    execute_bounded_literal_batch_source(&mut self.execution, source).map(CompiledGpuResource::new)
                },
            },
            _ => Err(invalid("compiled source task does not reference a source plan node")
                .with_fact("compiled_task_id", task.id())
                .with_fact("plan_node_id", node_id)
                .with_fact("pipeline_task_kind", task.kind().as_str())),
        }
        .map_err(|error| attach_task_facts(error, task, node_id))?;
        Ok(vec![(output, resource)])
    }

    /// Counts down one scheduled read of a GPU-resident input and drops the
    /// handle after the last one. The frame returned by that read keeps its
    /// own alias of the device buffers, so releasing the handle here frees
    /// nothing a later task still needs; it only lets the caller's input stop
    /// being pinned by this run.
    fn retire_gpu_input_read(&mut self, input_name: &str) -> Result<()> {
        let remaining = self.remaining_gpu_input_reads.get_mut(input_name).ok_or_else(|| {
            invalid("compiled resident source has no remaining input read").with_fact("input_name", input_name)
        })?;
        *remaining -= 1;
        if *remaining == 0 {
            self.remaining_gpu_input_reads.remove(input_name);
            self.gpu_inputs.remove(input_name);
        }
        Ok(())
    }

    fn execute_local_parquet_source(
        &mut self,
        task: &PipelineTask,
        node_id: NodeId,
        output: ResidentResourceId,
        source: &LocalParquetSource,
    ) -> Result<CompiledGpuResource> {
        let TaskPhysical::SourceRead(read) = task.physical() else {
            return Err(invalid("compiled local Parquet source task has no source read plan")
                .with_fact("compiled_task_id", task.id())
                .with_fact("pipeline_task_kind", task.kind().as_str()));
        };
        let shared = self.resource_sharing(output) != ResourceSharing::SingleConsume;
        // An aggregate consuming this stream may run one chunk on each of the run's lanes at once,
        // so the chunk target counts the regions of all of them. A run without lanes keeps one
        // read ahead of a streamed source instead, which the target counts as one more slot,
        // unless an aggregate consumer may take the one path that reads no chunk ahead.
        let aggregate_lane_width = self.aggregate_dispatch.as_ref().map_or(1, AggregateLaneDispatch::width);
        let read_ahead = source_read_ahead_applies(&self.execution);
        let aggregate_read_ahead = read_ahead
            && self.consuming_aggregates(task.id()).iter().all(|(aggregate, has_source_adjacent_stages)| {
                aggregate_source_stream_reads_ahead(aggregate, *has_source_adjacent_stages)
            });
        let context = self.execution.context();
        let metrics = self.execution.metrics();
        let (stream, join_leaf_adjacent_stages) = match read.output {
            SourceOutput::Frame => {
                return execute_bounded_local_parquet_source(&mut self.execution, source, node_id)
                    .map(CompiledGpuResource::new);
            },
            SourceOutput::JoinLeafFrame { adjacent_stages } => {
                // The materialized leaf keeps every chunk until it concatenates them, so no read
                // ahead adds to its live chunks.
                let mut stream = plan_bounded_join_leaf_local_parquet_source_stream(
                    context,
                    source,
                    node_id,
                    adjacent_stages,
                    false,
                    metrics,
                )?;
                // Route producers precede this task through compiled control edges, so a
                // sole consumer's routes are ready here. Shared frames keep their routes
                // for per-consumer post-decode application.
                if !shared {
                    stream = self.apply_dynamic_filters_to_source_stream(stream)?;
                }
                return materialize_bounded_join_leaf_local_parquet_source_stream(&mut self.execution, stream)
                    .map(CompiledGpuResource::new);
            },
            SourceOutput::JoinLeafStream { adjacent_stages } => (
                plan_bounded_join_leaf_local_parquet_source_stream(
                    context,
                    source,
                    node_id,
                    adjacent_stages,
                    read_ahead,
                    metrics,
                )?,
                Some(adjacent_stages),
            ),
            SourceOutput::AggregateInputStream { adjacent_stages, chunk_hints } => (
                plan_bounded_aggregate_input_local_parquet_source_stream(
                    context,
                    source,
                    node_id,
                    adjacent_stages,
                    chunk_hints,
                    aggregate_lane_width,
                    aggregate_read_ahead,
                    metrics,
                )?,
                None,
            ),
            SourceOutput::SharedJoinAggregateStream { adjacent_stages, chunk_hints } => (
                plan_bounded_shared_join_aggregate_local_parquet_source_stream(
                    context,
                    source,
                    node_id,
                    adjacent_stages,
                    chunk_hints,
                    aggregate_lane_width,
                    aggregate_read_ahead,
                    metrics,
                )?,
                None,
            ),
            // The unary consumers append their stages, and the last one
            // settles the stream into the frame its consumers take.
            SourceOutput::UnaryChainStream => {
                let stream = plan_bounded_unary_chain_local_parquet_source_stream(context, source, node_id, metrics)?;
                return Ok(CompiledGpuResource::join_leaf_local_parquet_source(stream));
            },
        };
        // A prefetch starts the read now on a pool worker under an attempt
        // auxiliary permit and parks it as a pending payload; the consumer
        // finishes it later. `None` means no permit or budget was free, and
        // the read stays deferred without error.
        if read.prefetch_eligible
            && let Some(prefetch) =
                try_start_bounded_join_leaf_local_parquet_source_prefetch(&mut self.execution, &stream)?
        {
            return Ok(CompiledGpuResource::pending_join_leaf_local_parquet_source(prefetch));
        }
        // A one-chunk leaf with no stages, no sharing and no filter routes
        // gains nothing from staying deferred; reading it here spares the
        // consumer a stream it would materialize at once anyway.
        if join_leaf_adjacent_stages == Some(false)
            && !shared
            && stream.chunk_count() <= 1
            && !read.dynamic_filter_routes
        {
            return materialize_bounded_join_leaf_local_parquet_source_stream(&mut self.execution, stream)
                .map(CompiledGpuResource::new);
        }
        Ok(CompiledGpuResource::join_leaf_local_parquet_source(stream))
    }

    /// The aggregates the task's output reaches through unary stages, each with whether a
    /// unary stage sits between the source and that aggregate.
    fn consuming_aggregates(&self, task: PipelineTaskId) -> Vec<(&AggregateNode, bool)> {
        let mut aggregates = Vec::new();
        let mut pending = vec![(task, false)];
        let mut visited = BTreeSet::new();
        while let Some((current, through_unary)) = pending.pop() {
            if !visited.insert((current, through_unary)) {
                continue;
            }
            for edge in self.pipeline.edges().iter().filter(|edge| {
                edge.from() == current
                    && matches!(edge.kind(), PipelineEdgeKind::ResidentFrame | PipelineEdgeKind::SharedResidentProducer)
            }) {
                let Some(consumer) = self.pipeline.tasks().iter().find(|consumer| consumer.id() == edge.to()) else {
                    continue;
                };
                match consumer.kind() {
                    PipelineTaskKind::Aggregate => {
                        if let Some(PlanNode::Aggregate(aggregate)) =
                            consumer.plan_node().and_then(|node| self.plan.node(node)).map(|node| &node.kind)
                        {
                            aggregates.push((aggregate, through_unary));
                        }
                    },
                    PipelineTaskKind::Unary => pending.push((edge.to(), true)),
                    _ => {},
                }
            }
        }
        aggregates
    }
}
