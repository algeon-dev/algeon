//! Adaptive bounded-pipeline admission facade.
//!
//! Adaptive admission evidence and reports remain separate from strict bounded
//! admission. Runtime execution remains in `exec::bounded::adaptive`.

use super::shared::{CapabilityFirstCandidateStatus, capability_first_bounded_pipeline_admission};
use crate::{
    adaptive_report::{
        AdaptiveAdmissionRuntimeEvidence, AdaptiveCandidateStatus, AdaptiveMemoryHistoryEvidence,
        AdaptivePipelineReport, AdaptiveRuntimeActionsAllowed, AdaptiveStrategy,
    },
    adaptive_runtime::AdaptiveMemoryHistory,
    capability::StrictBoundedEnvelope,
    error::Result,
    mode::{AdaptiveExecutionMode, NativePipelineExecutionMode},
    pipeline::CompiledPipeline,
    plan::QueryPlan,
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

    let admission =
        capability_first_bounded_pipeline_admission(plan)?.expect("join plan should produce admission evidence");
    let runtime_evidence = runtime_evidence_for_plan(plan, memory_history, admission.envelope());
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
        admission.rejection_projection(),
    )?;
    Ok(Some(AdaptivePipelineAdmission { report, pipeline }))
}

fn runtime_evidence_for_plan(
    plan: &QueryPlan,
    memory_history: Option<&AdaptiveMemoryHistory>,
    envelope: Option<StrictBoundedEnvelope>,
) -> AdaptiveAdmissionRuntimeEvidence {
    let _ = envelope;
    let mut runtime_actions_allowed = AdaptiveRuntimeActionsAllowed::default();
    let memory_history_evidence = match memory_history {
        Some(memory_history) => {
            runtime_actions_allowed.memory_history_lookup = true;
            memory_history
                .lookup_query_plan(plan)
                .map_or(AdaptiveMemoryHistoryEvidence::LookupMiss, AdaptiveMemoryHistoryEvidence::Observed)
        },
        None => AdaptiveMemoryHistoryEvidence::NotUsed,
    };
    AdaptiveAdmissionRuntimeEvidence { runtime_actions_allowed, memory_history_evidence }
}
