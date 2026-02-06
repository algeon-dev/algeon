//! Adaptive bounded-pipeline execution facade.
//!
//! Phase E.3 adds adaptive admission and report rows. Real adaptive runtime
//! actions belong to later Phase E steps and must remain separate from strict
//! bounded contract reporting.

use super::*;
use crate::{
    adaptive_report::{AdaptiveCandidateStatus, AdaptivePipelineReport, AdaptiveStrategy},
    mode::{AdaptiveExecutionMode, NativePipelineExecutionMode},
    pipeline::{CompiledPipeline, compile_pipeline},
};

#[derive(Debug, Clone)]
pub struct AdaptivePipelineAdmission {
    report: AdaptivePipelineReport,
    pipeline: Option<CompiledPipeline>,
}

impl AdaptivePipelineAdmission {
    pub fn execution_mode(&self) -> AdaptiveExecutionMode {
        self.report.execution_mode()
    }

    pub fn native_mode(&self) -> NativePipelineExecutionMode {
        self.execution_mode().native_mode()
    }

    pub fn report(&self) -> &AdaptivePipelineReport {
        &self.report
    }

    pub fn pipeline(&self) -> Option<&CompiledPipeline> {
        self.pipeline.as_ref()
    }
}

pub(super) fn try_execute_adaptive_bounded_pipeline(
    context: &ExecutionContext,
    plan: &QueryPlan,
    gpu_inputs: &GpuResidentInputMap,
    metrics: &mut QueryMetrics,
) -> Result<Option<QueryOutput>> {
    let Some(admission) = adaptive_bounded_pipeline_admission(plan)? else {
        return Ok(None);
    };
    execute_adaptive_bounded_pipeline(context, plan, gpu_inputs, metrics, admission).map(Some)
}

pub fn adaptive_bounded_pipeline_admission(plan: &QueryPlan) -> Result<Option<AdaptivePipelineAdmission>> {
    if !plan.contains_join() {
        return Ok(None);
    }

    let analysis = plan.capability_analysis();
    let (candidate_status, adaptive_strategy, rejection_reason, pipeline) =
        if analysis.has_rejections() || !analysis.admits_strict_bounded_join() {
            (AdaptiveCandidateStatus::Rejected, AdaptiveStrategy::Unsupported, Some("missing_strict_contracts"), None)
        } else if !matches!(plan.sink(), ResultSink::HostArrow { .. }) {
            (AdaptiveCandidateStatus::Rejected, AdaptiveStrategy::Unsupported, Some("unsupported_sink_boundary"), None)
        } else {
            let pipeline = compile_pipeline(plan, &analysis)?;
            match compiled_bounded_query_plan_support(plan, &pipeline)? {
                CompiledBoundedPipelineSupport::Supported => (
                    AdaptiveCandidateStatus::Admitted,
                    AdaptiveStrategy::ConservativeCompiledPipeline,
                    None,
                    Some(pipeline),
                ),
                CompiledBoundedPipelineSupport::Unsupported => (
                    AdaptiveCandidateStatus::Rejected,
                    AdaptiveStrategy::Unsupported,
                    Some("unsupported_compiled_pipeline_envelope"),
                    None,
                ),
            }
        };
    let report = AdaptivePipelineReport::from_admission_parts(
        &analysis,
        plan,
        candidate_status,
        adaptive_strategy,
        rejection_reason,
    )?;
    Ok(Some(AdaptivePipelineAdmission { report, pipeline }))
}

fn execute_adaptive_bounded_pipeline(
    _context: &ExecutionContext,
    _plan: &QueryPlan,
    _gpu_inputs: &GpuResidentInputMap,
    _metrics: &mut QueryMetrics,
    admission: AdaptivePipelineAdmission,
) -> Result<QueryOutput> {
    debug_assert_eq!(admission.native_mode(), NativePipelineExecutionMode::Adaptive);
    Err(QueryError::new(
        QueryErrorCode::UnsupportedPlanShape,
        "cudf_query_engine.bounded.adaptive_executor_unavailable",
        "adaptive bounded pipeline execution is not implemented in Phase E.3",
    )
    .with_fact("execution_mode", admission.execution_mode().as_str())
    .with_fact("candidate_status", admission.report().row().candidate_status.as_str())
    .with_fact("strict_bounded_contract_report", "not_emitted"))
}
