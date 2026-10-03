use query_planning::{
    error::Result,
    pipeline::{PipelineTask, ResidentResourceId},
    plan::PlanNode,
};

use super::{CompiledGpuResource, CompiledGpuTaskExecutor, CompiledPipelineState};
use crate::exec::{
    bounded::{
        compiled::{
            frame_stream::{map_filter_resident_frame_stream, map_projection_resident_frame_stream},
            helpers::{attach_task_facts, invalid, plan_node, single_output_resource},
            resource::CompiledGpuResourcePayload,
        },
        execute_bounded_projection, run_budgeted,
    },
    operators::execute_filter_with_budget,
};

impl CompiledGpuTaskExecutor<'_> {
    pub(super) fn execute_unary_task(
        &mut self,
        task: &PipelineTask,
        state: &mut CompiledPipelineState<CompiledGpuResource>,
    ) -> Result<Vec<(ResidentResourceId, CompiledGpuResource)>> {
        let node_id = plan_node(task)?;
        let input = *task.input_resources().first().ok_or_else(|| {
            invalid("compiled unary task has no input resource")
                .with_fact("compiled_task_id", task.id())
                .with_fact("pipeline_task_kind", task.kind().as_str())
        })?;
        let output = single_output_resource(task)?;
        let node = self.plan.node(node_id).ok_or_else(|| {
            invalid("compiled unary task references a missing plan node")
                .with_fact("compiled_task_id", task.id())
                .with_fact("plan_node_id", node_id)
        })?;
        if !matches!(node.kind, PlanNode::Filter(_) | PlanNode::Projection(_)) {
            return Err(invalid("compiled unary task does not reference a filter or projection plan node")
                .with_fact("compiled_task_id", task.id())
                .with_fact("plan_node_id", node_id)
                .with_fact("pipeline_task_kind", task.kind().as_str()));
        }
        let input_payload = self.take_task_input(task, state, input)?;
        let resource = match (&node.kind, input_payload) {
            (PlanNode::Filter(filter), CompiledGpuResourcePayload::JoinLeafLocalParquetSource(stream)) => Ok(
                CompiledGpuResource::join_leaf_local_parquet_source(stream.with_filter_stage(node_id, filter.clone())),
            ),
            (PlanNode::Filter(filter), CompiledGpuResourcePayload::ResidentFrame { frame, .. }) => self
                .apply_dynamic_filters_to_filter_frame(node_id, filter, *frame)
                .and_then(|frame| {
                    run_budgeted(
                        &mut self.execution,
                        node_id,
                        "compiled_filter",
                        frame.frame,
                        |context, input, token, metrics| {
                            execute_filter_with_budget(context, filter, input, token, metrics)
                        },
                    )
                })
                .map(CompiledGpuResource::new),
            (PlanNode::Filter(filter), CompiledGpuResourcePayload::ResidentFrameStream(stream)) => {
                self.apply_dynamic_filters_to_filter_stream(node_id, filter, stream).map(|stream| {
                    CompiledGpuResource::frame_stream_payload(map_filter_resident_frame_stream(
                        node_id,
                        stream,
                        filter.clone(),
                    ))
                })
            },
            (PlanNode::Projection(projection), CompiledGpuResourcePayload::JoinLeafLocalParquetSource(stream)) => {
                Ok(CompiledGpuResource::join_leaf_local_parquet_source(
                    stream.with_projection_stage(node_id, projection.clone()),
                ))
            },
            (PlanNode::Projection(projection), CompiledGpuResourcePayload::ResidentFrame { frame, .. }) => {
                execute_bounded_projection(&mut self.execution, node_id, "compiled_projection", projection, *frame)
                    .map(CompiledGpuResource::new)
            },
            (PlanNode::Projection(projection), CompiledGpuResourcePayload::ResidentFrameStream(stream)) => {
                Ok(CompiledGpuResource::frame_stream_payload(map_projection_resident_frame_stream(
                    node_id,
                    stream,
                    projection.clone(),
                )))
            },
            (_, payload) => Err(self.reject_task_input(task, input, payload)?),
        }
        .map_err(|error| attach_task_facts(error, task, node_id))?;
        Ok(vec![(output, resource)])
    }
}
