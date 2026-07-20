//! Adaptive bounded-pipeline execution facade.

use super::{compiled::execute_compiled_pipeline, *};
use crate::{
    adaptive_report::{AdaptiveCandidateStatus, AdaptivePipelineReport, AdaptivePipelineReportRow, AdaptiveStrategy},
    admission::{AdaptivePipelineAdmission, adaptive_bounded_pipeline_admission_with_memory_history},
    mode::NativePipelineExecutionMode,
};

pub(super) fn try_execute_adaptive_bounded_pipeline(
    context: &ExecutionContext,
    plan: &QueryPlan,
    gpu_inputs: &GpuResidentInputMap,
    metrics: &mut QueryMetrics,
    observation_scope: Option<&crate::observability::RuntimeExecutionObservationScope>,
) -> Result<Option<QueryOutput>> {
    let memory_history = context.adaptive_memory_history();
    let Some(admission) = adaptive_bounded_pipeline_admission_with_memory_history(plan, memory_history.as_deref())?
    else {
        return Ok(None);
    };
    execute_adaptive_bounded_pipeline(context, plan, gpu_inputs, metrics, admission, observation_scope).map(Some)
}

fn execute_adaptive_bounded_pipeline(
    context: &ExecutionContext,
    plan: &QueryPlan,
    gpu_inputs: &GpuResidentInputMap,
    metrics: &mut QueryMetrics,
    admission: AdaptivePipelineAdmission,
    observation_scope: Option<&crate::observability::RuntimeExecutionObservationScope>,
) -> Result<QueryOutput> {
    AdaptiveCompiledPipelineExecutor::new(context, plan, gpu_inputs, metrics, admission, observation_scope).run()
}

struct AdaptiveCompiledPipelineExecutor<'a, 'm> {
    context: &'a ExecutionContext,
    plan: &'a QueryPlan,
    gpu_inputs: &'a GpuResidentInputMap,
    metrics: &'m mut QueryMetrics,
    admission: AdaptivePipelineAdmission,
    observation_scope: Option<&'a crate::observability::RuntimeExecutionObservationScope>,
}

impl<'a, 'm> AdaptiveCompiledPipelineExecutor<'a, 'm> {
    fn new(
        context: &'a ExecutionContext,
        plan: &'a QueryPlan,
        gpu_inputs: &'a GpuResidentInputMap,
        metrics: &'m mut QueryMetrics,
        admission: AdaptivePipelineAdmission,
        observation_scope: Option<&'a crate::observability::RuntimeExecutionObservationScope>,
    ) -> Self {
        Self { context, plan, gpu_inputs, metrics, admission, observation_scope }
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
                        "nexus_query_engine.bounded.adaptive_executor",
                        "admitted adaptive candidate must include a compiled pipeline",
                    )
                    .with_fact("execution_mode", self.admission.execution_mode().as_str())
                    .with_fact("candidate_status", AdaptiveCandidateStatus::Admitted.as_str())
                })?;
                match self.admission.report().row().adaptive_strategy {
                    AdaptiveStrategy::ConservativeCompiledPipeline => {
                        let output = execute_compiled_pipeline(
                            self.context,
                            self.plan,
                            pipeline,
                            self.gpu_inputs,
                            self.metrics,
                            self.observation_scope,
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
    QueryError::new(QueryErrorCode::UnsupportedPlanShape, "nexus_query_engine.bounded.adaptive_executor", message)
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
