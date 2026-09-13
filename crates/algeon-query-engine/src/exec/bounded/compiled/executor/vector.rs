#[cfg(feature = "cuvs")]
use std::collections::BTreeMap;

use super::{CompiledGpuResource, CompiledGpuTaskExecutor, CompiledPipelineState};
#[cfg(feature = "cuvs")]
use crate::exec::bounded::compiled::helpers::attach_task_facts;
#[cfg(feature = "cuvs")]
use crate::exec::bounded::compiled::resource::CompiledGpuResourcePayload;
use crate::{
    error::Result,
    exec::bounded::compiled::helpers::{invalid, plan_node, single_output_resource},
    pipeline::{PipelineTask, ResidentResourceId},
};
#[cfg(not(feature = "cuvs"))]
use crate::{exec::vector_algorithm_execution_unavailable, plan::PlanNode};
#[cfg(feature = "cuvs")]
use crate::{
    exec::{
        bounded::ResidentGpuFrame, execute_vector_algorithm, observe_device_memory_bytes,
        validate_vector_algorithm_execution,
    },
    plan::PlanNode,
};

impl CompiledGpuTaskExecutor<'_> {
    pub(super) fn execute_vector_algorithm_task(
        &mut self,
        task: &PipelineTask,
        state: &mut CompiledPipelineState<CompiledGpuResource>,
    ) -> Result<Vec<(ResidentResourceId, CompiledGpuResource)>> {
        let node_id = plan_node(task)?;
        let output = single_output_resource(task)?;
        let node = self.plan.node(node_id).ok_or_else(|| {
            invalid("compiled vector task references a missing plan node")
                .with_fact("compiled_task_id", task.id())
                .with_fact("plan_node_id", node_id)
        })?;
        let PlanNode::VectorAlgorithm(vector) = &node.kind else {
            return Err(invalid("compiled vector task does not reference a vector algorithm node")
                .with_fact("compiled_task_id", task.id())
                .with_fact("plan_node_id", node_id));
        };

        #[cfg(not(feature = "cuvs"))]
        {
            let _ = (output, state);
            return Err(vector_algorithm_execution_unavailable(node_id, vector));
        }

        #[cfg(feature = "cuvs")]
        {
            validate_vector_algorithm_execution(node_id, vector)?;
            if vector.inputs.len() != task.input_resources().len() {
                return Err(invalid("compiled vector task input resources do not match vector inputs")
                    .with_fact("compiled_task_id", task.id())
                    .with_fact("plan_node_id", node_id)
                    .with_fact("vector_input_count", vector.inputs.len())
                    .with_fact("resource_count", task.input_resources().len()));
            }

            let mut inputs = BTreeMap::new();
            for (input_id, resource) in vector.inputs.iter().copied().zip(task.input_resources().iter().copied()) {
                if inputs.contains_key(&input_id) {
                    continue;
                }
                let frame = if state.can_consume_task_input_resource(task, resource) {
                    let payload = state.consume_task_input_resource(task, resource)?;
                    let CompiledGpuResourcePayload::ResidentFrame { frame, .. } = payload.payload else {
                        return Err(invalid("compiled vector task requires resident frame inputs")
                            .with_fact("compiled_task_id", task.id())
                            .with_fact("plan_node_id", node_id)
                            .with_fact("resident_resource_id", resource));
                    };
                    let ResidentGpuFrame { frame, .. } = *frame;
                    frame
                } else {
                    let payload = state.resource_payload(resource)?;
                    let CompiledGpuResourcePayload::ResidentFrame { frame, .. } = &payload.payload else {
                        return Err(invalid("shared compiled vector input must be a resident frame")
                            .with_fact("compiled_task_id", task.id())
                            .with_fact("plan_node_id", node_id)
                            .with_fact("resident_resource_id", resource));
                    };
                    frame.frame.clone()
                };
                inputs.insert(input_id, frame);
            }
            let frame = execute_vector_algorithm(self.execution.context(), self.plan, node_id, vector, &inputs)
                .map_err(|error| attach_task_facts(error, task, node_id))?;
            let device_bytes = observe_device_memory_bytes(&frame, self.execution.metrics())
                .map_err(|error| attach_task_facts(error, task, node_id))?;
            Ok(vec![(output, CompiledGpuResource::new(ResidentGpuFrame { frame, device_bytes }))])
        }
    }
}
