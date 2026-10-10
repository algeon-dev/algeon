#[cfg(not(feature = "cuvs"))]
use query_planning::plan::PlanNode;
#[cfg(feature = "cuvs")]
use query_planning::plan::PlanNode;
use query_planning::{
    error::Result,
    pipeline::{PipelineTask, ResidentResourceId},
};

use super::{CompiledGpuResource, CompiledGpuTaskExecutor, CompiledPipelineState};
#[cfg(feature = "cuvs")]
use crate::exec::bounded::compiled::helpers::attach_task_facts;
use crate::exec::bounded::compiled::helpers::{invalid, plan_node, single_output_resource};
#[cfg(not(feature = "cuvs"))]
use crate::exec::vector_algorithm_execution_unavailable;
#[cfg(feature = "cuvs")]
use crate::{
    exec::bounded::ResidentGpuFrame, exec::execute_vector_algorithm, exec::observe_device_memory_bytes,
    exec::validate_vector_algorithm_execution,
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
            Err(vector_algorithm_execution_unavailable(node_id, vector))
        }

        #[cfg(feature = "cuvs")]
        {
            validate_vector_algorithm_execution(node_id, vector)?;
            let inputs = self.collect_dense_relation_inputs(task, state, node_id, &vector.inputs)?;
            let frame = execute_vector_algorithm(self.execution.context(), self.plan, node_id, vector, &inputs)
                .map_err(|error| attach_task_facts(error, task, node_id))?;
            let device_bytes = observe_device_memory_bytes(&frame, self.execution.metrics())
                .map_err(|error| attach_task_facts(error, task, node_id))?;
            Ok(vec![(output, CompiledGpuResource::new(ResidentGpuFrame { frame, device_bytes }))])
        }
    }
}
