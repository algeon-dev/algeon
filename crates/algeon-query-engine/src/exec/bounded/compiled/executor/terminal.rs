use query_planning::{
    error::Result,
    pipeline::{PipelineTask, ResidentDataForm, ResidentResourceId, ResourceSharing, SinkBoundaryKind},
    plan::{NodeId, PlanNode, ResultSink},
};

use super::{CompiledGpuResource, CompiledGpuTaskExecutor, CompiledPipelineState};
use crate::exec::{
    bounded::{
        ResidentGpuFrame,
        compiled::{
            helpers::{
                attach_task_facts, invalid, limit_node, plan_node, single_input_resource, single_output_resource,
            },
            resource::CompiledGpuResourcePayload,
        },
        execute_bounded_unary, execute_bounded_union, materialize_bounded_frame_to_host,
        union_branch_with_output_names,
    },
    operators::execute_limit,
};

impl CompiledGpuTaskExecutor<'_> {
    pub(super) fn execute_limit_task(
        &mut self,
        task: &PipelineTask,
        state: &mut CompiledPipelineState<CompiledGpuResource>,
    ) -> Result<Vec<(ResidentResourceId, CompiledGpuResource)>> {
        let node_id = plan_node(task)?;
        let input = single_input_resource(task)?;
        let output = single_output_resource(task)?;
        let limit = limit_node(self.plan, task, node_id)?;
        let input_frame = match self.take_task_input(task, state, input)? {
            // The fused top-k already applied this limit, offset included;
            // applying it again would skip the offset rows twice.
            CompiledGpuResourcePayload::ResidentFrame { frame, applied_limit } if applied_limit == Some(node_id) => {
                return Ok(vec![(output, CompiledGpuResource::new(*frame))]);
            },
            CompiledGpuResourcePayload::ResidentFrame { frame, .. } => *frame,
            payload => return Err(self.reject_task_input(task, input, payload)?),
        };
        let frame =
            execute_bounded_unary(&mut self.execution, node_id, input_frame.frame, |context, input, metrics| {
                execute_limit(context, limit, input, metrics)
            })
            .map_err(|error| attach_task_facts(error, task, node_id))?;
        Ok(vec![(output, CompiledGpuResource::new(frame))])
    }

    pub(super) fn execute_union_task(
        &mut self,
        task: &PipelineTask,
        state: &mut CompiledPipelineState<CompiledGpuResource>,
    ) -> Result<Vec<(ResidentResourceId, CompiledGpuResource)>> {
        let node_id = plan_node(task)?;
        let output = single_output_resource(task)?;
        let Some(node) = self.plan.node(node_id) else {
            return Err(invalid("compiled union task references a missing plan node")
                .with_fact("compiled_task_id", task.id())
                .with_fact("plan_node_id", node_id));
        };
        if !matches!(node.kind, PlanNode::Union(_)) {
            return Err(invalid("compiled union task does not reference a union plan node")
                .with_fact("compiled_task_id", task.id())
                .with_fact("plan_node_id", node_id)
                .with_fact("pipeline_task_kind", task.kind().as_str()));
        }
        let inputs = self.union_task_inputs(task, state, node_id)?;
        // When the compiled contract lets this output stay a stream for its single consumer,
        // chain the branch frames without a union-level concat copy. Stream consumers pull in
        // branch order, and exact-order consumers drain in order before their own required
        // concat, so neither branch order nor summation order changes. The union step itself
        // never holds the branch inputs plus a full copy at once; a downstream whole-input
        // consumer can still concatenate afterwards on its own contract, which keeps its own
        // peak accounting.
        if !inputs.is_empty()
            && self.resource_data_form(output) == ResidentDataForm::FrameStream
            && self.resource_sharing(output) == ResourceSharing::SingleConsume
        {
            return Ok(vec![(output, CompiledGpuResource::frame_stream(inputs))]);
        }
        let frame = execute_bounded_union(&mut self.execution, node_id, inputs)
            .map_err(|error| attach_task_facts(error, task, node_id))?;
        Ok(vec![(output, CompiledGpuResource::new(frame))])
    }

    pub(super) fn union_task_inputs(
        &mut self,
        task: &PipelineTask,
        state: &mut CompiledPipelineState<CompiledGpuResource>,
        node_id: NodeId,
    ) -> Result<Vec<ResidentGpuFrame>> {
        let mut inputs = Vec::new();
        for resource in task.input_resources() {
            inputs.push(self.take_frame_input(task, state, *resource)?);
        }
        let Some(node) = self.plan.node(node_id) else {
            return Err(invalid("compiled union task references a missing plan node")
                .with_fact("compiled_task_id", task.id())
                .with_fact("plan_node_id", node_id));
        };
        inputs.into_iter().map(|input| union_branch_with_output_names(node_id, input, node.output_schema())).collect()
    }

    pub(super) fn execute_sink_materialize_task(
        &mut self,
        task: &PipelineTask,
        state: &mut CompiledPipelineState<CompiledGpuResource>,
    ) -> Result<Vec<(ResidentResourceId, CompiledGpuResource)>> {
        if self.pipeline.sink().kind() != SinkBoundaryKind::HostArrowMaterialization
            || self.pipeline.sink().input() != task.id()
        {
            return Err(invalid("compiled sink materialize task does not match the pipeline sink boundary")
                .with_fact("compiled_task_id", task.id())
                .with_fact("pipeline_task_kind", task.kind().as_str())
                .with_fact("sink_boundary_task_id", self.pipeline.sink().input())
                .with_fact("sink_boundary_kind", self.pipeline.sink().kind().as_str()));
        }
        if self.host_output.is_some() {
            return Err(invalid("compiled host sink materialized more than once")
                .with_fact("compiled_task_id", task.id())
                .with_fact("pipeline_task_kind", task.kind().as_str()));
        }
        let input = single_input_resource(task)?;
        let ResultSink::HostArrow { input: final_node_id, schema } = self.plan.sink() else {
            return Err(invalid("compiled sink materialize task requires a host Arrow sink")
                .with_fact("compiled_task_id", task.id())
                .with_fact("pipeline_task_kind", task.kind().as_str()));
        };
        let executable_schema = self
            .plan
            .node(*final_node_id)
            .ok_or_else(|| {
                invalid("compiled sink task references a missing final plan node")
                    .with_fact("compiled_task_id", task.id())
                    .with_fact("plan_node_id", final_node_id)
            })?
            .output_schema();
        let input_frame = self.take_frame_input(task, state, input)?;
        let output = materialize_bounded_frame_to_host(
            &mut self.execution,
            input_frame.frame,
            input_frame.device_bytes,
            *final_node_id,
            executable_schema,
            schema,
        )
        .map_err(|error| {
            error
                .with_fact_if_absent("compiled_task_id", task.id())
                .with_fact_if_absent("pipeline_task_kind", task.kind().as_str())
                .with_fact_if_absent("resident_resource_id", input)
                .with_fact_if_absent("plan_node_id", final_node_id)
                .with_fact_if_absent("sink_boundary_kind", self.pipeline.sink().kind().as_str())
        })?;
        // The host batch is not a resident resource; it leaves through
        // `take_host_output` once the scheduler returns.
        self.host_output = Some(output);
        Ok(Vec::new())
    }
}
