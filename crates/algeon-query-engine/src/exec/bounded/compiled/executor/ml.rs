use query_planning::{
    error::Result,
    pipeline::{PipelineTask, ResidentResourceId},
    plan::PlanNode,
};

use super::{CompiledGpuResource, CompiledGpuTaskExecutor, CompiledPipelineState};
use crate::exec::bounded::compiled::helpers::{invalid, plan_node};
#[cfg(not(feature = "cuml"))]
use crate::exec::ml_algorithm_execution_unavailable;
#[cfg(feature = "cuml")]
use crate::{
    exec::bounded::compiled::helpers::{attach_task_facts, single_output_resource},
    exec::{
        bounded::ResidentGpuFrame,
        ml_algorithm::{execute_ml_algorithm, validate_ml_algorithm_execution},
        observe_device_memory_bytes,
    },
};

impl CompiledGpuTaskExecutor<'_> {
    pub(super) fn execute_ml_algorithm_task(
        &mut self,
        task: &PipelineTask,
        state: &mut CompiledPipelineState<CompiledGpuResource>,
    ) -> Result<Vec<(ResidentResourceId, CompiledGpuResource)>> {
        let node_id = plan_node(task)?;
        let node = self.plan.node(node_id).ok_or_else(|| invalid("compiled ML task references a missing plan node"))?;
        let PlanNode::MlAlgorithm(ml) = &node.kind else {
            return Err(invalid("compiled ML task must reference an ML algorithm node"));
        };
        #[cfg(not(feature = "cuml"))]
        {
            let _ = state;
            Err(ml_algorithm_execution_unavailable(node_id, ml))
        }
        #[cfg(feature = "cuml")]
        {
            let output = single_output_resource(task)?;
            validate_ml_algorithm_execution(self.plan, node_id, ml)?;
            let inputs = self.collect_dense_relation_inputs(task, state, node_id, &ml.inputs)?;
            let frame = execute_ml_algorithm(self.execution.context(), self.plan, node_id, ml, &inputs)
                .map_err(|error| attach_task_facts(error, task, node_id))?;
            let device_bytes = observe_device_memory_bytes(&frame, self.execution.metrics())
                .map_err(|error| attach_task_facts(error, task, node_id))?;
            Ok(vec![(output, CompiledGpuResource::new(ResidentGpuFrame { frame, device_bytes }))])
        }
    }
}
