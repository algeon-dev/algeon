//! Adaptive bounded-pipeline execution facade.
//!
//! Phase E.5 adds the adaptive executor shell. The initial strategy is a
//! conservative compiled-pipeline wrapper with adaptive labels and action
//! accounting; richer runtime actions belong to later Phase E steps.

use super::{compiled::execute_compiled_pipeline_to_host, *};
use crate::{
    adaptive_report::{AdaptiveCandidateStatus, AdaptivePipelineReport, AdaptivePipelineReportRow, AdaptiveStrategy},
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

    pub fn runtime_action_counts(&self) -> crate::adaptive_report::AdaptiveRuntimeActionCounts {
        self.report.row().runtime_action_counts
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
    context: &ExecutionContext,
    plan: &QueryPlan,
    gpu_inputs: &GpuResidentInputMap,
    metrics: &mut QueryMetrics,
    admission: AdaptivePipelineAdmission,
) -> Result<QueryOutput> {
    AdaptiveCompiledPipelineExecutor::new(context, plan, gpu_inputs, metrics, admission).run()
}

struct AdaptiveCompiledPipelineExecutor<'a, 'm> {
    context: &'a ExecutionContext,
    plan: &'a QueryPlan,
    gpu_inputs: &'a GpuResidentInputMap,
    metrics: &'m mut QueryMetrics,
    admission: AdaptivePipelineAdmission,
}

impl<'a, 'm> AdaptiveCompiledPipelineExecutor<'a, 'm> {
    fn new(
        context: &'a ExecutionContext,
        plan: &'a QueryPlan,
        gpu_inputs: &'a GpuResidentInputMap,
        metrics: &'m mut QueryMetrics,
        admission: AdaptivePipelineAdmission,
    ) -> Self {
        Self { context, plan, gpu_inputs, metrics, admission }
    }

    fn run(self) -> Result<QueryOutput> {
        debug_assert_eq!(self.admission.native_mode(), NativePipelineExecutionMode::Adaptive);
        let counts = self.admission.runtime_action_counts();
        self.metrics.record_adaptive_runtime_action_counts(
            counts.retry_count,
            counts.spill_count,
            counts.downgrade_count,
        );

        match self.admission.report().row().candidate_status {
            AdaptiveCandidateStatus::Rejected => Err(unsupported_adaptive_candidate_error(self.admission.report())),
            AdaptiveCandidateStatus::Admitted => {
                let pipeline = self.admission.pipeline().ok_or_else(|| {
                    QueryError::new(
                        QueryErrorCode::InvalidPlan,
                        "cudf_query_engine.bounded.adaptive_executor",
                        "admitted adaptive candidate must include a compiled pipeline",
                    )
                    .with_fact("execution_mode", self.admission.execution_mode().as_str())
                    .with_fact("candidate_status", AdaptiveCandidateStatus::Admitted.as_str())
                })?;
                match self.admission.report().row().adaptive_strategy {
                    AdaptiveStrategy::ConservativeCompiledPipeline => execute_compiled_pipeline_to_host(
                        self.context,
                        self.plan,
                        pipeline,
                        self.gpu_inputs,
                        self.metrics,
                    ),
                    AdaptiveStrategy::Unsupported => Err(unsupported_adaptive_candidate_error(self.admission.report())),
                }
            },
        }
    }
}

fn unsupported_adaptive_candidate_error(report: &AdaptivePipelineReport) -> QueryError {
    let row = report.row();
    adaptive_candidate_error("adaptive bounded pipeline candidate is unsupported by the adaptive executor", row)
}

fn adaptive_candidate_error(message: &'static str, row: &AdaptivePipelineReportRow) -> QueryError {
    QueryError::new(QueryErrorCode::UnsupportedPlanShape, "cudf_query_engine.bounded.adaptive_executor", message)
        .with_fact("execution_mode", row.execution_mode.as_str())
        .with_fact("candidate_status", row.candidate_status.as_str())
        .with_fact("adaptive_strategy", row.adaptive_strategy.as_str())
        .with_fact("runtime_actions_allowed", row.runtime_actions_allowed.to_stable_value())
        .with_fact("memory_history_evidence", row.memory_history_evidence.as_str())
        .with_fact("retry_count", row.runtime_action_counts.retry_count)
        .with_fact("spill_count", row.runtime_action_counts.spill_count)
        .with_fact("downgrade_count", row.runtime_action_counts.downgrade_count)
        .with_fact("missing_strict_contract_count", row.missing_strict_contracts.len())
        .with_fact("rejection_reason", row.rejection_reason.unwrap_or("none"))
        .with_fact("strict_bounded_contract_report", "not_emitted")
        .with_fact("adaptive_report_row", row.to_stable_kv_row())
}
