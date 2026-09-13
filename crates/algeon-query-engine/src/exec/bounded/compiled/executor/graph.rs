#[cfg(feature = "cugraph")]
use std::collections::BTreeMap;

use super::{CompiledGpuResource, CompiledGpuTaskExecutor, CompiledPipelineState, CompiledTaskOutput};
#[cfg(feature = "cugraph")]
use crate::exec::bounded::compiled::{
    helpers::{attach_task_facts, plan_node, single_output_resource},
    resource::CompiledGpuResourcePayload,
};
#[cfg(feature = "cugraph")]
use crate::exec::bounded::execute_bounded_graph_algorithm_from_inputs;
use crate::{error::Result, exec::bounded::compiled::helpers::invalid, pipeline::PipelineTask};
#[cfg(feature = "cugraph")]
use crate::{
    exec::bounded::ResidentGpuFrame,
    plan::{NodeId, PlanNode},
};

impl CompiledGpuTaskExecutor<'_> {
    #[cfg(feature = "cugraph")]
    pub(super) fn execute_graph_algorithm_task(
        &mut self,
        task: &PipelineTask,
        state: &mut CompiledPipelineState<CompiledGpuResource>,
    ) -> Result<CompiledTaskOutput<CompiledGpuResource>> {
        let node_id = plan_node(task)?;
        let output = single_output_resource(task)?;
        let Some(node) = self.plan.node(node_id) else {
            return Err(invalid("compiled graph task references a missing plan node")
                .with_fact("compiled_task_id", task.id())
                .with_fact("plan_node_id", node_id));
        };
        let PlanNode::GraphAlgorithm(graph) = &node.kind else {
            return Err(invalid("compiled graph task does not reference a graph algorithm node")
                .with_fact("compiled_task_id", task.id())
                .with_fact("plan_node_id", node_id)
                .with_fact("pipeline_task_kind", task.kind().as_str()));
        };
        if graph.inputs.len() != task.input_resources().len() {
            return Err(invalid("compiled graph task input resources do not match graph inputs")
                .with_fact("compiled_task_id", task.id())
                .with_fact("plan_node_id", node_id)
                .with_fact("graph_input_count", graph.inputs.len())
                .with_fact("resource_count", task.input_resources().len()));
        }

        let mut inputs = BTreeMap::new();
        for (input_id, resource) in graph.inputs.iter().copied().zip(task.input_resources().iter().copied()) {
            let frame = if state.can_consume_task_input_resource(task, resource) {
                let payload = state.consume_task_input_resource(task, resource)?;
                let CompiledGpuResourcePayload::ResidentFrame { frame, .. } = payload.payload else {
                    self.release_graph_task_inputs(inputs);
                    return Err(invalid("compiled graph task requires resident frame inputs")
                        .with_fact("compiled_task_id", task.id())
                        .with_fact("plan_node_id", node_id)
                        .with_fact("resident_resource_id", resource));
                };
                *frame
            } else {
                let payload = state.resource_payload(resource)?;
                let CompiledGpuResourcePayload::ResidentFrame { frame, .. } = &payload.payload else {
                    self.release_graph_task_inputs(inputs);
                    return Err(invalid("shared compiled graph input must be a resident frame")
                        .with_fact("compiled_task_id", task.id())
                        .with_fact("plan_node_id", node_id)
                        .with_fact("resident_resource_id", resource));
                };
                ResidentGpuFrame { frame: frame.frame.clone(), device_bytes: frame.device_bytes() }
            };
            inputs.insert(input_id, frame);
        }
        let Some(edge) = inputs.remove(&graph.input) else {
            self.release_graph_task_inputs(inputs);
            return Err(invalid("compiled graph task is missing its edge input")
                .with_fact("compiled_task_id", task.id())
                .with_fact("plan_node_id", node_id)
                .with_fact("edge_input", graph.input));
        };
        let frame = execute_bounded_graph_algorithm_from_inputs(&mut self.execution, node_id, graph, edge, inputs)
            .map_err(|error| attach_task_facts(error, task, node_id))?;
        Ok(CompiledTaskOutput::new(vec![(output, CompiledGpuResource::new(frame))]))
    }

    #[cfg(not(feature = "cugraph"))]
    pub(super) fn execute_graph_algorithm_task(
        &mut self,
        task: &PipelineTask,
        _state: &mut CompiledPipelineState<CompiledGpuResource>,
    ) -> Result<CompiledTaskOutput<CompiledGpuResource>> {
        Err(invalid("compiled graph task requires the cugraph feature")
            .with_fact("compiled_task_id", task.id())
            .with_fact("operator", "graph_algorithm"))
    }

    #[cfg(feature = "cugraph")]
    fn release_graph_task_inputs(&mut self, _inputs: BTreeMap<NodeId, ResidentGpuFrame>) {}
}
