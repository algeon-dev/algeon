//! Strict bounded-pipeline execution boundary.
//!
//! This module is the strict-only facade for compiled bounded execution. Public
//! and crate-local strict exports should pass through this boundary instead of
//! importing non-strict admission or report surfaces.

pub(in crate::exec::bounded) use super::compiled::join_strategy::CompiledJoinExecutionStrategy;
#[doc(hidden)]
pub use super::compiled::{
    CompiledBoundedPipelineSupport, CompiledGpuResource, CompiledGpuTaskExecutor, CompiledPipelineScheduler,
    CompiledPipelineState, CompiledTaskAdmission, CompiledTaskExecutor, CompiledTaskOutput, ReleasedCompiledResource,
    ResidentFrameStreamProducerTestHarness, YieldedFrameCounterTestHarness, compiled_bounded_pipeline_support,
    compiled_bounded_query_plan_support,
};
use super::{
    admission::{
        CapabilityFirstCandidateStatus, CapabilityFirstPipelineAdmission, capability_first_bounded_pipeline_admission,
    },
    compiled::execute_compiled_pipeline_to_host,
};
use crate::{
    capability::CapabilityAnalysis,
    error::{QueryError, QueryErrorCode, Result},
    exec::types::{GpuResidentInputMap, QueryOutput},
    metrics::QueryMetrics,
    mode::NativePipelineExecutionMode,
    pipeline::CompiledPipeline,
    plan::{QueryPlan, ResultSink},
    runtime::ExecutionContext,
};

/// Strict bounded admission evidence for a compiled bounded pipeline.
///
/// Presence means capability analysis admitted the plan and compilation
/// produced the exact pipeline that strict execution will run.
#[derive(Debug, Clone)]
pub struct StrictBoundedPipelineAdmission {
    analysis: CapabilityAnalysis,
    pipeline: CompiledPipeline,
}

impl StrictBoundedPipelineAdmission {
    /// Capability facts used to admit the plan.
    pub fn analysis(&self) -> &CapabilityAnalysis {
        &self.analysis
    }

    /// Compiled pipeline admitted for strict bounded execution.
    pub fn pipeline(&self) -> &CompiledPipeline {
        &self.pipeline
    }

    /// Native pipeline mode represented by this admission.
    pub const fn native_mode(&self) -> NativePipelineExecutionMode {
        NativePipelineExecutionMode::StrictBounded
    }
}

/// Run strict bounded admission for a query plan.
///
/// `Ok(None)` means the plan is outside strict compiled-pipeline support.
/// `Ok(Some(_))` means every strict gate required for compilation was modeled.
pub fn strict_bounded_pipeline_admission(plan: &QueryPlan) -> Result<Option<StrictBoundedPipelineAdmission>> {
    let Some(admission) = capability_first_bounded_pipeline_admission(plan)? else {
        return Ok(None);
    };
    match admission.candidate_status() {
        CapabilityFirstCandidateStatus::Admitted => strict_admission_from_capability_first(admission).map(Some),
        CapabilityFirstCandidateStatus::Rejected => match admission.rejection_reason() {
            Some("unsupported_sink_boundary") | Some("unsupported_compiled_pipeline_envelope") => {
                Err(compiled_bounded_pipeline_required_error(
                    plan,
                    admission.pipeline(),
                    admission.rejection_reason().unwrap_or("unknown"),
                ))
            },
            _ => Ok(None),
        },
    }
}

fn strict_admission_from_capability_first(
    admission: CapabilityFirstPipelineAdmission,
) -> Result<StrictBoundedPipelineAdmission> {
    let (analysis, pipeline) = admission.into_parts();
    let pipeline = pipeline.ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.bounded.strict_admission",
            "admitted strict bounded candidate must include a compiled pipeline",
        )
        .with_fact("execution_mode", NativePipelineExecutionMode::StrictBounded.as_str())
        .with_fact("candidate_status", "admitted")
    })?;
    Ok(StrictBoundedPipelineAdmission { analysis, pipeline })
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
