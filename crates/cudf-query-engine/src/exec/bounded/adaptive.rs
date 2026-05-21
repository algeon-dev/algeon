//! Adaptive bounded-pipeline execution facade.
//!
//! Phase E.5 adds the adaptive executor shell. The initial strategy is a
//! conservative compiled-pipeline wrapper with adaptive labels and action
//! accounting; richer runtime actions belong to later Phase E steps.

use super::{
    admission::{CapabilityFirstCandidateStatus, capability_first_bounded_pipeline_admission},
    compiled::execute_compiled_pipeline_to_host,
    *,
};
use crate::{
    adaptive_report::{
        AdaptiveAdmissionRuntimeEvidence, AdaptiveCandidateStatus, AdaptiveMemoryHistoryEvidence,
        AdaptivePipelineReport, AdaptivePipelineReportRow, AdaptiveRuntimeActionsAllowed, AdaptiveStrategy,
    },
    adaptive_runtime::AdaptiveMemoryHistory,
    mode::{AdaptiveExecutionMode, NativePipelineExecutionMode},
    pipeline::CompiledPipeline,
};

/// Adaptive bounded admission evidence for a join-capable query plan.
///
/// The report is always present. A compiled pipeline is present only when
/// capability-first admission accepted the plan for adaptive execution.
#[derive(Debug, Clone)]
pub struct AdaptivePipelineAdmission {
    report: AdaptivePipelineReport,
    pipeline: Option<CompiledPipeline>,
}

impl AdaptivePipelineAdmission {
    /// Adaptive execution mode recorded in the admission report.
    pub fn execution_mode(&self) -> AdaptiveExecutionMode {
        self.report.execution_mode()
    }

    /// Native pipeline mode represented by this admission.
    pub fn native_mode(&self) -> NativePipelineExecutionMode {
        self.execution_mode().native_mode()
    }

    /// Structured adaptive report used by diagnostics and report rows.
    pub fn report(&self) -> &AdaptivePipelineReport {
        &self.report
    }

    /// Compiled pipeline to execute when the candidate was admitted.
    pub fn pipeline(&self) -> Option<&CompiledPipeline> {
        self.pipeline.as_ref()
    }

    /// Runtime action counters promised by the adaptive strategy.
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
    let memory_history = context.adaptive_memory_history();
    let Some(admission) = adaptive_bounded_pipeline_admission_with_memory_history(plan, memory_history.as_deref())?
    else {
        return Ok(None);
    };
    execute_adaptive_bounded_pipeline(context, plan, gpu_inputs, metrics, admission).map(Some)
}

/// Run adaptive bounded admission without historical memory evidence.
pub fn adaptive_bounded_pipeline_admission(plan: &QueryPlan) -> Result<Option<AdaptivePipelineAdmission>> {
    adaptive_bounded_pipeline_admission_with_memory_history(plan, None)
}

/// Run adaptive bounded admission with optional memory-history evidence.
///
/// Non-join plans return `Ok(None)`. Join plans return a report whether they
/// are admitted or rejected so callers can keep adaptive evidence separate
/// from strict bounded contract rows.
pub fn adaptive_bounded_pipeline_admission_with_memory_history(
    plan: &QueryPlan,
    memory_history: Option<&AdaptiveMemoryHistory>,
) -> Result<Option<AdaptivePipelineAdmission>> {
    if !plan.contains_join() {
        return Ok(None);
    }

    let runtime_evidence = runtime_evidence_for_plan(plan, memory_history);
    let admission =
        capability_first_bounded_pipeline_admission(plan)?.expect("join plan should produce admission evidence");
    let (candidate_status, adaptive_strategy, rejection_reason, pipeline) = match admission.candidate_status() {
        CapabilityFirstCandidateStatus::Admitted => (
            AdaptiveCandidateStatus::Admitted,
            AdaptiveStrategy::ConservativeCompiledPipeline,
            None,
            admission.pipeline().cloned(),
        ),
        CapabilityFirstCandidateStatus::Rejected => {
            (AdaptiveCandidateStatus::Rejected, AdaptiveStrategy::Unsupported, admission.rejection_reason(), None)
        },
    };
    let report = AdaptivePipelineReport::from_admission_parts(
        admission.analysis(),
        plan,
        candidate_status,
        adaptive_strategy,
        runtime_evidence,
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
        let memory_history_evidence = &self.admission.report().row().memory_history_evidence;
        if memory_history_evidence.lookup_attempted() {
            self.metrics.record_adaptive_memory_history_lookup(memory_history_evidence.lookup_hit());
        }

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
                    AdaptiveStrategy::ConservativeCompiledPipeline => {
                        let output = execute_compiled_pipeline_to_host(
                            self.context,
                            self.plan,
                            pipeline,
                            self.gpu_inputs,
                            self.metrics,
                        )?;
                        if let Some(memory_history) = self.context.adaptive_memory_history() {
                            memory_history.record_observation(
                                crate::adaptive_runtime::AdaptiveMemoryHistoryKey::for_query_plan(self.plan),
                                crate::adaptive_runtime::AdaptiveMemoryObservation::from_query_metrics(self.metrics),
                            );
                            memory_history.record_task_observations_from_query_metrics(
                                self.plan,
                                pipeline,
                                self.metrics,
                            );
                        }
                        Ok(output)
                    },
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
        .with_fact("memory_history_evidence", row.memory_history_evidence.to_stable_value())
        .with_fact("retry_count", row.runtime_action_counts.retry_count)
        .with_fact("spill_count", row.runtime_action_counts.spill_count)
        .with_fact("downgrade_count", row.runtime_action_counts.downgrade_count)
        .with_fact("missing_strict_contract_count", row.missing_strict_contracts.len())
        .with_fact("rejection_reason", row.rejection_reason.unwrap_or("none"))
        .with_fact("strict_bounded_contract_report", "not_emitted")
        .with_fact("adaptive_report_row", row.to_stable_kv_row())
}

fn runtime_evidence_for_plan(
    plan: &QueryPlan,
    memory_history: Option<&AdaptiveMemoryHistory>,
) -> AdaptiveAdmissionRuntimeEvidence {
    let Some(memory_history) = memory_history else {
        return AdaptiveAdmissionRuntimeEvidence::default();
    };

    let memory_history_evidence = memory_history
        .lookup_query_plan(plan)
        .map_or(AdaptiveMemoryHistoryEvidence::LookupMiss, AdaptiveMemoryHistoryEvidence::Observed);
    AdaptiveAdmissionRuntimeEvidence {
        runtime_actions_allowed: AdaptiveRuntimeActionsAllowed {
            memory_history_lookup: true,
            ..AdaptiveRuntimeActionsAllowed::default()
        },
        memory_history_evidence,
    }
}
