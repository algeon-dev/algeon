//! Bounded-pipeline execution boundary (the single capability-first path).

use super::compiled::execute_compiled_pipeline;
#[doc(hidden)]
pub use super::compiled::{
    CompiledGpuResource, CompiledGpuTaskExecutor, CompiledPipelineScheduler, CompiledPipelineState,
    CompiledSchedulingCycle, CompiledTaskAdmission, CompiledTaskExecutor, CompiledTaskOutput, ReleasedCompiledResource,
};
#[cfg(test)]
pub(crate) use super::compiled::{ResidentFrameStreamProducerTestHarness, YieldedFrameCounterTestHarness};
use crate::{
    admission::bounded_pipeline_admission_with_statistical_aggregate_mode,
    error::Result,
    exec::output::QueryOutput,
    metrics::QueryMetrics,
    plan::QueryPlan,
    runtime::{ExecutionContext, resident::GpuResidentInputMap},
};

pub(in crate::exec) fn try_execute_bounded_pipeline(
    context: &ExecutionContext,
    plan: &QueryPlan,
    gpu_inputs: &GpuResidentInputMap,
    metrics: &mut QueryMetrics,
    observation_scope: Option<&crate::observability::RuntimeExecutionObservationScope>,
) -> Result<Option<QueryOutput>> {
    let Some(admission) = bounded_pipeline_admission_with_statistical_aggregate_mode(
        plan,
        context.statistical_aggregate_execution_mode(),
    )?
    else {
        return Ok(None);
    };
    execute_compiled_pipeline(context, plan, admission.pipeline(), gpu_inputs, metrics, observation_scope).map(Some)
}
