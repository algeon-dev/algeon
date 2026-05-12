//! Capability-first bounded-pipeline admission shared by strict and adaptive facades.

use super::compiled::{CompiledBoundedPipelineSupport, compiled_bounded_query_plan_support};
use crate::{
    capability::CapabilityAnalysis,
    error::Result,
    pipeline::{CompiledPipeline, compile_pipeline},
    plan::{QueryPlan, ResultSink},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::exec::bounded) enum CapabilityFirstCandidateStatus {
    Admitted,
    Rejected,
}

#[derive(Debug, Clone)]
pub(in crate::exec::bounded) struct CapabilityFirstPipelineAdmission {
    analysis: CapabilityAnalysis,
    pipeline: Option<CompiledPipeline>,
    candidate_status: CapabilityFirstCandidateStatus,
    rejection_reason: Option<&'static str>,
}

impl CapabilityFirstPipelineAdmission {
    pub(in crate::exec::bounded) fn analysis(&self) -> &CapabilityAnalysis {
        &self.analysis
    }

    pub(in crate::exec::bounded) fn pipeline(&self) -> Option<&CompiledPipeline> {
        self.pipeline.as_ref()
    }

    pub(in crate::exec::bounded) fn candidate_status(&self) -> CapabilityFirstCandidateStatus {
        self.candidate_status
    }

    pub(in crate::exec::bounded) fn rejection_reason(&self) -> Option<&'static str> {
        self.rejection_reason
    }

    pub(in crate::exec::bounded) fn into_parts(self) -> (CapabilityAnalysis, Option<CompiledPipeline>) {
        (self.analysis, self.pipeline)
    }

    fn admitted(analysis: CapabilityAnalysis, pipeline: CompiledPipeline) -> Self {
        Self {
            analysis,
            pipeline: Some(pipeline),
            candidate_status: CapabilityFirstCandidateStatus::Admitted,
            rejection_reason: None,
        }
    }

    fn rejected(
        analysis: CapabilityAnalysis,
        pipeline: Option<CompiledPipeline>,
        rejection_reason: &'static str,
    ) -> Self {
        Self {
            analysis,
            pipeline,
            candidate_status: CapabilityFirstCandidateStatus::Rejected,
            rejection_reason: Some(rejection_reason),
        }
    }
}

pub(in crate::exec::bounded) fn capability_first_bounded_pipeline_admission(
    plan: &QueryPlan,
) -> Result<Option<CapabilityFirstPipelineAdmission>> {
    if !plan.contains_join() {
        return Ok(None);
    }

    let analysis = plan.capability_analysis();
    if analysis.has_rejections() {
        return Ok(Some(CapabilityFirstPipelineAdmission::rejected(analysis, None, "missing_strict_contracts")));
    }
    if !matches!(plan.sink(), ResultSink::HostArrow { .. }) {
        return Ok(Some(CapabilityFirstPipelineAdmission::rejected(analysis, None, "unsupported_sink_boundary")));
    }

    let pipeline = compile_pipeline(plan, &analysis)?;
    match compiled_bounded_query_plan_support(plan, &pipeline)? {
        CompiledBoundedPipelineSupport::Supported => {
            Ok(Some(CapabilityFirstPipelineAdmission::admitted(analysis, pipeline)))
        },
        CompiledBoundedPipelineSupport::Unsupported => Ok(Some(CapabilityFirstPipelineAdmission::rejected(
            analysis,
            Some(pipeline),
            "unsupported_compiled_pipeline_envelope",
        ))),
    }
}
