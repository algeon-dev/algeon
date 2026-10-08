use query_planning::{
    error::Result,
    pipeline::{PipelineTask, ResidentResourceId, ResourceSharing, SourceOutput, TaskPhysical},
    plan::{NodeId, PlanNode},
    source::{LocalParquetSource, SourceNode},
};

use super::{CompiledGpuResource, CompiledGpuTaskExecutor};
use crate::exec::bounded::{
    compiled::helpers::{attach_task_facts, invalid, plan_node, single_output_resource},
    execute_bounded_gpu_resident_source, execute_bounded_literal_batch_source, execute_bounded_local_parquet_source,
    materialize_bounded_join_leaf_local_parquet_source_stream,
    plan_bounded_aggregate_input_local_parquet_source_stream, plan_bounded_join_leaf_local_parquet_source_stream,
    plan_bounded_shared_join_aggregate_local_parquet_source_stream,
    plan_bounded_unary_chain_local_parquet_source_stream, try_start_bounded_join_leaf_local_parquet_source_prefetch,
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
        let context = self.execution.context();
        let metrics = self.execution.metrics();
        let (stream, join_leaf_adjacent_stages) = match read.output {
            SourceOutput::Frame => {
                return execute_bounded_local_parquet_source(&mut self.execution, source, node_id)
                    .map(CompiledGpuResource::new);
            },
            SourceOutput::JoinLeafFrame { adjacent_stages } => {
                let mut stream = plan_bounded_join_leaf_local_parquet_source_stream(
                    context,
                    source,
                    node_id,
                    adjacent_stages,
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
                plan_bounded_join_leaf_local_parquet_source_stream(context, source, node_id, adjacent_stages, metrics)?,
                Some(adjacent_stages),
            ),
            SourceOutput::AggregateInputStream { adjacent_stages, chunk_hints } => (
                plan_bounded_aggregate_input_local_parquet_source_stream(
                    context,
                    source,
                    node_id,
                    adjacent_stages,
                    chunk_hints,
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
        if read.prefetch_eligible
            && let Some(prefetch) =
                try_start_bounded_join_leaf_local_parquet_source_prefetch(&mut self.execution, &stream)?
        {
            return Ok(CompiledGpuResource::pending_join_leaf_local_parquet_source(prefetch));
        }
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
}
