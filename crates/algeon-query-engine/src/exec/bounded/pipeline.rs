//! Bounded-pipeline execution boundary (the single capability-first path).

#[cfg(test)]
pub(crate) use super::compiled::YieldedFrameCounterTestHarness;
use super::compiled::execute_compiled_pipeline;
#[cfg(feature = "test-hooks")]
#[doc(hidden)]
pub use super::compiled::{
    CompiledGpuResource, CompiledGpuTaskExecutor, CompiledPipelineScheduler, CompiledPipelineState,
    CompiledSchedulingCycle, CompiledTaskExecutor, CompiledTaskOutput, LaneDispatchSupport, ReleasedCompiledResource,
    is_lane_family_pipeline, run_lane_dispatched,
};
use crate::{
    admission::{BoundedPipelineAdmission, artifact_binding_error, bounded_pipeline_decision},
    error::Result,
    exec::{ExecutionMetrics, output::QueryOutput},
    plan::QueryPlan,
    runtime::{ExecutionContext, resident::GpuResidentInputMap},
};

pub(in crate::exec) fn try_execute_bounded_pipeline(
    context: &ExecutionContext,
    plan: &QueryPlan,
    admission: Option<&BoundedPipelineAdmission>,
    gpu_inputs: &GpuResidentInputMap,
    metrics: &ExecutionMetrics,
    observation_scope: Option<&crate::observability::RuntimeExecutionObservationScope>,
) -> Result<Option<QueryOutput>> {
    if let Some(admission) = admission {
        // A supplied artifact is checked, never re-decided: a plan, mode, or
        // sink the artifact does not bind to is a caller error, not a fallback.
        if let Some(projection) = admission.binding_rejection(plan, context.statistical_aggregate_execution_mode()) {
            return Err(artifact_binding_error(plan, Some(admission.pipeline()), &projection));
        }
        return execute_compiled_pipeline(context, plan, admission.pipeline(), gpu_inputs, metrics, observation_scope)
            .map(Some);
    }
    let admission = bounded_pipeline_decision(
        plan,
        context.statistical_aggregate_execution_mode(),
        crate::rapids_backend::rapids_analysis_context(),
    )?
    .into_execution_admission()?;
    let Some(admission) = admission else {
        return Ok(None);
    };
    execute_compiled_pipeline(context, plan, admission.pipeline(), gpu_inputs, metrics, observation_scope).map(Some)
}
