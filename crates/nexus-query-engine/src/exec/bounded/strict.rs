//! Strict bounded-pipeline execution boundary.

use super::compiled::execute_compiled_pipeline;
#[doc(hidden)]
pub use super::compiled::{
    CompiledGpuResource, CompiledGpuTaskExecutor, CompiledPipelineScheduler, CompiledPipelineState,
    CompiledSchedulingCycle, CompiledTaskAdmission, CompiledTaskExecutor, CompiledTaskOutput, ReleasedCompiledResource,
    ResidentFrameStreamProducerTestHarness, YieldedFrameCounterTestHarness,
};
use crate::{
    admission::strict_bounded_pipeline_admission_with_statistical_aggregate_mode,
    error::Result,
    exec::output::QueryOutput,
    metrics::QueryMetrics,
    mode::NativePipelineExecutionMode,
    plan::QueryPlan,
    runtime::{ExecutionContext, resident::GpuResidentInputMap},
};

pub(in crate::exec) fn try_execute_strict_bounded_pipeline(
    context: &ExecutionContext,
    plan: &QueryPlan,
    gpu_inputs: &GpuResidentInputMap,
    metrics: &mut QueryMetrics,
) -> Result<Option<QueryOutput>> {
    let Some(admission) = strict_bounded_pipeline_admission_with_statistical_aggregate_mode(
        plan,
        context.statistical_aggregate_execution_mode(),
    )?
    else {
        return Ok(None);
    };
    debug_assert_eq!(admission.native_mode(), NativePipelineExecutionMode::StrictBounded);
    execute_compiled_pipeline(context, plan, admission.pipeline(), gpu_inputs, metrics).map(Some)
}
