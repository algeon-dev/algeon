//! Bounded-pipeline execution boundary (the single capability-first path).

#[cfg(test)]
pub(crate) use super::compiled::YieldedFrameCounterTestHarness;
use super::compiled::execute_compiled_pipeline;
#[cfg(feature = "test-hooks")]
#[doc(hidden)]
pub use super::compiled::{
    CompiledGpuResource, CompiledGpuTaskExecutor, CompiledPipelineScheduler, CompiledPipelineState,
    CompiledTaskExecutor, LaneDispatchSupport, is_lane_family_pipeline, run_lane_dispatched,
};
use crate::{
    admission::AdmittedPlan,
    error::Result,
    exec::{ExecutionMetrics, output::QueryOutput},
    observability::RuntimeExecutionObservationScope,
    runtime::{ExecutionContext, resident::GpuResidentInputMap},
};

pub(in crate::exec) fn execute_bounded_pipeline(
    context: &ExecutionContext,
    admitted: &AdmittedPlan,
    gpu_inputs: &GpuResidentInputMap,
    metrics: &ExecutionMetrics,
    observation_scope: Option<&RuntimeExecutionObservationScope>,
) -> Result<QueryOutput> {
    execute_compiled_pipeline(
        context,
        admitted.plan(),
        admitted.policy(),
        admitted.pipeline(),
        gpu_inputs,
        metrics,
        observation_scope,
    )
}
