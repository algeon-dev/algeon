//! Strict bounded-pipeline execution boundary.
//!
//! This module is the strict-only facade for compiled bounded execution. Public
//! and crate-local strict exports should pass through this boundary instead of
//! importing non-strict admission or report surfaces.

use super::compiled::execute_compiled_pipeline_to_host;
pub(in crate::exec::bounded) use super::compiled::join_strategy::CompiledJoinExecutionStrategy;
#[doc(hidden)]
pub use super::compiled::{
    CompiledBoundedPipelineSupport, CompiledGpuResource, CompiledGpuTaskExecutor, CompiledPipelineScheduler,
    CompiledPipelineState, CompiledTaskExecutor, CompiledTaskOutput, ReleasedCompiledResource,
    compiled_bounded_pipeline_support, compiled_bounded_query_plan_support,
};
use crate::{
    capability::CapabilityAnalysis,
    error::{QueryError, QueryErrorCode, Result},
    exec::types::{GpuResidentInputMap, QueryOutput},
    metrics::QueryMetrics,
    mode::NativePipelineExecutionMode,
    pipeline::{CompiledPipeline, compile_pipeline},
    plan::{QueryPlan, ResultSink},
    runtime::ExecutionContext,
};

#[derive(Debug, Clone)]
pub struct StrictBoundedPipelineAdmission {
    analysis: CapabilityAnalysis,
    pipeline: CompiledPipeline,
}

impl StrictBoundedPipelineAdmission {
    pub fn analysis(&self) -> &CapabilityAnalysis {
        &self.analysis
    }

    pub fn pipeline(&self) -> &CompiledPipeline {
        &self.pipeline
    }

    pub const fn native_mode(&self) -> NativePipelineExecutionMode {
        NativePipelineExecutionMode::StrictBounded
    }
}

pub fn strict_bounded_pipeline_admission(plan: &QueryPlan) -> Result<Option<StrictBoundedPipelineAdmission>> {
    if !plan.contains_join() {
        return Ok(None);
    }

    let analysis = plan.capability_analysis();
    if analysis.has_rejections() {
        return Ok(None);
    }
    if !matches!(plan.sink(), ResultSink::HostArrow { .. }) {
        return Err(compiled_bounded_pipeline_required_error(plan, None, "unsupported_sink_boundary"));
    }

    let pipeline = compile_pipeline(plan, &analysis)?;
    match compiled_bounded_query_plan_support(plan, &pipeline)? {
        CompiledBoundedPipelineSupport::Unsupported => Err(compiled_bounded_pipeline_required_error(
            plan,
            Some(&pipeline),
            "unsupported_compiled_pipeline_envelope",
        )),
        CompiledBoundedPipelineSupport::Supported => Ok(Some(StrictBoundedPipelineAdmission { analysis, pipeline })),
    }
}

pub(in crate::exec) fn try_execute_strict_bounded_pipeline(
    context: &ExecutionContext,
    plan: &QueryPlan,
    gpu_inputs: &GpuResidentInputMap,
    metrics: &mut QueryMetrics,
) -> Result<Option<QueryOutput>> {
    let Some(admission) = strict_bounded_pipeline_admission(plan)? else {
        return Ok(None);
    };
    debug_assert_eq!(admission.native_mode(), NativePipelineExecutionMode::StrictBounded);
    execute_compiled_pipeline_to_host(context, plan, admission.pipeline(), gpu_inputs, metrics).map(Some)
}

fn compiled_bounded_pipeline_required_error(
    plan: &QueryPlan,
    pipeline: Option<&crate::pipeline::CompiledPipeline>,
    reason: &'static str,
) -> QueryError {
    let sink_kind = match plan.sink() {
        ResultSink::GpuResident { .. } => "gpu_resident",
        ResultSink::HostArrow { .. } => "host_arrow",
    };
    let mut error = QueryError::new(
        QueryErrorCode::UnsupportedPlanShape,
        "cudf_query_engine.bounded.compiled_pipeline_required",
        "capability-admitted bounded join execution requires the compiled bounded pipeline runtime",
    )
    .with_fact("compiled_pipeline_required", "true")
    .with_fact("compiled_pipeline_rejection_reason", reason)
    .with_fact("legacy_bounded_join_path_bridge", "disabled")
    .with_fact("result_sink_kind", sink_kind);

    if let Some(pipeline) = pipeline {
        error = error
            .with_fact("compiled_task_count", pipeline.tasks().len())
            .with_fact("compiled_edge_count", pipeline.edges().len())
            .with_fact("compiled_barrier_count", pipeline.barriers().len())
            .with_fact("compiled_resource_count", pipeline.resources().len())
            .with_fact("compiled_gate_count", pipeline.gates().len())
            .with_fact("compiled_sink_kind", pipeline.sink().kind().as_str());
    }

    error
}
