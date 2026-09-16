//! Capability-first bounded-pipeline admission with one decision path.

use super::{CompiledBoundedPipelineSupport, compiled_bounded_query_plan_support};
use crate::{
    backend::AnalysisContext,
    capability::{CapabilityAnalysis, CapabilityEnvelope, CapabilityEnvelopeAdmission, CapabilityRejectionProjection},
    error::Result,
    pipeline::{CompiledPipeline, PipelineCompilation, PipelineTaskKind, ResidentDataForm, compile_pipeline_outcome},
    plan::QueryPlan,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CapabilityFirstCandidateStatus {
    Admitted,
    Rejected,
}

#[derive(Debug, Clone)]
pub(super) struct CapabilityFirstPipelineAdmission {
    analysis: CapabilityAnalysis,
    pipeline: Option<CompiledPipeline>,
    envelope: Option<CapabilityEnvelope>,
    rejection_projection: Option<CapabilityRejectionProjection>,
    candidate_status: CapabilityFirstCandidateStatus,
}

impl CapabilityFirstPipelineAdmission {
    pub(super) fn pipeline(&self) -> Option<&CompiledPipeline> {
        self.pipeline.as_ref()
    }

    pub(super) fn envelope(&self) -> Option<CapabilityEnvelope> {
        self.envelope
    }

    pub(super) fn candidate_status(&self) -> CapabilityFirstCandidateStatus {
        self.candidate_status
    }

    pub(super) fn rejection_projection(&self) -> Option<&CapabilityRejectionProjection> {
        self.rejection_projection.as_ref()
    }

    pub(super) fn into_parts(self) -> (CapabilityAnalysis, Option<CompiledPipeline>) {
        (self.analysis, self.pipeline)
    }

    fn admitted(analysis: CapabilityAnalysis, envelope: CapabilityEnvelope, pipeline: CompiledPipeline) -> Self {
        Self {
            analysis,
            pipeline: Some(pipeline),
            envelope: Some(envelope),
            rejection_projection: None,
            candidate_status: CapabilityFirstCandidateStatus::Admitted,
        }
    }

    fn rejected(
        analysis: CapabilityAnalysis,
        pipeline: Option<CompiledPipeline>,
        projection: CapabilityRejectionProjection,
    ) -> Self {
        Self {
            analysis,
            pipeline,
            envelope: None,
            rejection_projection: Some(projection),
            candidate_status: CapabilityFirstCandidateStatus::Rejected,
        }
    }
}

pub(super) fn capability_first_bounded_pipeline_admission(
    plan: &QueryPlan,
    ctx: AnalysisContext<'_>,
) -> Result<Option<CapabilityFirstPipelineAdmission>> {
    capability_first_bounded_pipeline_admission_impl(plan, ctx)
}

fn capability_first_bounded_pipeline_admission_impl(
    plan: &QueryPlan,
    ctx: AnalysisContext<'_>,
) -> Result<Option<CapabilityFirstPipelineAdmission>> {
    if !plan.contains_join() && !plan.contains_vector_algorithm() {
        return Ok(None);
    }

    let analysis = plan.capability_analysis(ctx);
    if let Some(projection) = analysis.first_local_rejection_projection() {
        return Ok(Some(CapabilityFirstPipelineAdmission::rejected(analysis, None, projection)));
    }
    let envelope_admission = analysis.capability_envelope_admission();
    let pipeline = match compile_pipeline_outcome(plan, &analysis)? {
        PipelineCompilation::Compiled(pipeline) => pipeline,
        // The compiler is the authority on constructability: when it cannot
        // build a pipeline, its structural reason (e.g. a dependency cycle)
        // reports the failure, not a plan-family envelope verdict about a
        // pipeline that does not exist.
        PipelineCompilation::Rejected(projection) => {
            return Ok(Some(CapabilityFirstPipelineAdmission::rejected(analysis, None, projection)));
        },
    };
    let envelope = if pipeline_is_within_migrated_execution_scope(&pipeline) {
        analysis.capability_envelope_label()
    } else {
        match envelope_admission {
            CapabilityEnvelopeAdmission::Admitted { envelope } => envelope,
            CapabilityEnvelopeAdmission::Rejected { projection } => {
                return Ok(Some(CapabilityFirstPipelineAdmission::rejected(analysis, Some(pipeline), projection)));
            },
        }
    };
    match compiled_bounded_query_plan_support(plan, &analysis, &pipeline)? {
        CompiledBoundedPipelineSupport::Supported => {
            Ok(Some(CapabilityFirstPipelineAdmission::admitted(analysis, envelope, pipeline)))
        },
        CompiledBoundedPipelineSupport::Unsupported(projection) => {
            Ok(Some(CapabilityFirstPipelineAdmission::rejected(analysis, Some(pipeline), projection)))
        },
    }
}

/// Execution capabilities the compiled task and resource contracts already
/// decide on their own.
///
/// Join task groups with unary, limit, final-sort, and aggregate stages are
/// admitted by those contracts alone: the named plan-family recognisers keep
/// no veto over such a pipeline and only supply its compatibility label. A
/// resource fanned out to several consumers is covered when it is produced by
/// a source read or is a complete immutable frame: frames are refcounted
/// snapshots, so every consumer borrows and releases them through the same
/// path as shared source reads, with no cursor to alias (dynamic-filter
/// application is a no-op for non-source frames, and the contract graph still
/// refuses unresolvable sharing at compile time). Everything else still needs
/// a capability no compiled contract carries yet — windowed and union
/// composition, shared frame streams, and which join kernels accept another
/// join's retained output — so it keeps the recogniser gate until those
/// contracts exist. Delete this function, and the branch that calls it, once
/// they do.
fn pipeline_is_within_migrated_execution_scope(pipeline: &CompiledPipeline) -> bool {
    pipeline.tasks().iter().filter(|task| task.kind() == PipelineTaskKind::JoinMaterialize).count() >= 1
        && pipeline.tasks().iter().all(|task| {
            matches!(
                task.kind(),
                PipelineTaskKind::SourceRead
                    | PipelineTaskKind::GpuResidentSource
                    | PipelineTaskKind::LiteralBatchSource
                    | PipelineTaskKind::Unary
                    | PipelineTaskKind::Limit
                    | PipelineTaskKind::FinalSort
                    | PipelineTaskKind::Aggregate
                    | PipelineTaskKind::JoinBuild
                    | PipelineTaskKind::JoinProbe
                    | PipelineTaskKind::JoinMaterialize
                    | PipelineTaskKind::SinkMaterialize
            )
        })
        && pipeline.resources().iter().all(|resource| {
            resource.consumers().len() <= 1
                || matches!(resource.data_form(), ResidentDataForm::Frame)
                || pipeline.tasks().iter().any(|task| {
                    task.id() == resource.producer()
                        && matches!(
                            task.kind(),
                            PipelineTaskKind::SourceRead
                                | PipelineTaskKind::GpuResidentSource
                                | PipelineTaskKind::LiteralBatchSource
                        )
                })
        })
}
