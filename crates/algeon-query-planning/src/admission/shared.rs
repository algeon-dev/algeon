//! Capability-first bounded-pipeline admission with one decision path.

use super::support::{CompiledBoundedPipelineSupport, compiled_bounded_query_plan_support};
use crate::{
    backend::AnalysisContext,
    capability::CapabilityRejectionProjection,
    error::Result,
    pipeline::{CompiledPipeline, PipelineCompilation, compile_pipeline_outcome},
    plan::QueryPlan,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CapabilityFirstCandidateStatus {
    Admitted,
    Rejected,
}

#[derive(Debug, Clone)]
pub(super) struct CapabilityFirstPipelineAdmission {
    pipeline: Option<CompiledPipeline>,
    rejection_projection: Option<CapabilityRejectionProjection>,
    candidate_status: CapabilityFirstCandidateStatus,
}

impl CapabilityFirstPipelineAdmission {
    pub(super) fn candidate_status(&self) -> CapabilityFirstCandidateStatus {
        self.candidate_status
    }

    pub(super) fn rejection_projection(&self) -> Option<&CapabilityRejectionProjection> {
        self.rejection_projection.as_ref()
    }

    pub(super) fn into_pipeline(self) -> Option<CompiledPipeline> {
        self.pipeline
    }

    fn admitted(pipeline: CompiledPipeline) -> Self {
        Self {
            pipeline: Some(pipeline),
            rejection_projection: None,
            candidate_status: CapabilityFirstCandidateStatus::Admitted,
        }
    }

    fn rejected(pipeline: Option<CompiledPipeline>, projection: CapabilityRejectionProjection) -> Self {
        Self {
            pipeline,
            rejection_projection: Some(projection),
            candidate_status: CapabilityFirstCandidateStatus::Rejected,
        }
    }
}

pub(super) fn capability_first_bounded_pipeline_admission(
    plan: &QueryPlan,
    ctx: AnalysisContext<'_>,
) -> Result<CapabilityFirstPipelineAdmission> {
    // One flow for every plan: capability analysis, pipeline compile, then the
    // task, resource, and join contracts. Whether a plan enters the compiled
    // pipeline depends on the operator capabilities its pipeline carries,
    // never on which plan shape surrounds them. A plan outside every contract
    // is rejected with the first contract it misses.
    let analysis = plan.capability_analysis(ctx);
    if let Some(projection) = analysis.first_rejection_projection() {
        return Ok(CapabilityFirstPipelineAdmission::rejected(None, projection));
    }
    let pipeline = match compile_pipeline_outcome(plan, &analysis)? {
        PipelineCompilation::Compiled(pipeline) => pipeline,
        // The compiler is the authority on constructability: when it cannot
        // build a pipeline, its structural reason (e.g. a dependency cycle)
        // reports the failure.
        PipelineCompilation::Rejected(projection) => {
            return Ok(CapabilityFirstPipelineAdmission::rejected(None, projection));
        },
    };
    match compiled_bounded_query_plan_support(plan, &analysis, &pipeline)? {
        CompiledBoundedPipelineSupport::Supported => Ok(CapabilityFirstPipelineAdmission::admitted(pipeline)),
        CompiledBoundedPipelineSupport::Unsupported(projection) => {
            Ok(CapabilityFirstPipelineAdmission::rejected(Some(pipeline), projection))
        },
    }
}
