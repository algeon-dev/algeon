use super::{CompiledGpuResource, CompiledGpuTaskExecutor, CompiledPipelineState, CompiledTaskOutput};
use crate::{
    error::Result,
    exec::bounded::{
        compiled::{
            helpers::{invalid, plan_node, single_input_resource, single_output_resource},
            resource::CompiledGpuResourcePayload,
        },
        execute_bounded_edge_normalize_from_input,
    },
    pipeline::PipelineTask,
    plan::PlanNode,
};

impl CompiledGpuTaskExecutor<'_> {
    pub(super) fn execute_edge_normalize_task(
        &mut self,
        task: &PipelineTask,
        state: &mut CompiledPipelineState<CompiledGpuResource>,
    ) -> Result<CompiledTaskOutput<CompiledGpuResource>> {
        let node_id = plan_node(task)?;
        let output = single_output_resource(task)?;
        let Some(node) = self.plan.node(node_id) else {
            return Err(invalid("compiled edge normalize task references a missing plan node")
                .with_fact("compiled_task_id", task.id())
                .with_fact("plan_node_id", node_id));
        };
        let PlanNode::EdgeNormalize(edge_normalize) = &node.kind else {
            return Err(invalid("compiled edge normalize task does not reference an edge normalize node")
                .with_fact("compiled_task_id", task.id())
                .with_fact("plan_node_id", node_id));
        };
        let input = single_input_resource(task)?;
        let payload = state.consume_task_input_resource(task, input)?.into_payload();
        let CompiledGpuResourcePayload::ResidentFrame { frame, .. } = payload.payload else {
            return Err(invalid("compiled edge normalize task requires a resident frame input")
                .with_fact("compiled_task_id", task.id())
                .with_fact("plan_node_id", node_id));
        };
        let frame = execute_bounded_edge_normalize_from_input(&mut self.execution, node_id, edge_normalize, *frame)?;
        Ok(CompiledTaskOutput::new(vec![(output, CompiledGpuResource::new(frame))]))
    }
}
