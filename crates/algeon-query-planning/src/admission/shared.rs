//! Capability-first bounded-pipeline admission with one decision path.

use super::{
    CompiledBoundedPipelineSupport, compiled_bounded_query_plan_support, pipeline_tasks_within_migrated_contract,
};
use crate::{
    backend::AnalysisContext,
    capability::{CapabilityEnvelope, CapabilityEnvelopeAdmission, CapabilityRejectionProjection},
    error::Result,
    pipeline::{
        CompiledPipeline, PipelineCompilation, PipelineTaskKind, ResidentDataForm, ResidentResource,
        compile_pipeline_outcome,
    },
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

    pub(super) fn into_pipeline(self) -> Option<CompiledPipeline> {
        self.pipeline
    }

    fn admitted(envelope: CapabilityEnvelope, pipeline: CompiledPipeline) -> Self {
        Self {
            pipeline: Some(pipeline),
            envelope: Some(envelope),
            rejection_projection: None,
            candidate_status: CapabilityFirstCandidateStatus::Admitted,
        }
    }

    fn rejected(pipeline: Option<CompiledPipeline>, projection: CapabilityRejectionProjection) -> Self {
        Self {
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
    // One flow for every plan family: capability analysis, pipeline compile,
    // then the shared task/resource support contracts. Whether a plan enters
    // the compiled pipeline depends on the operator capabilities its pipeline
    // carries, never on which plan shape surrounds them. Plans outside the
    // migrated contract keep the previous behavior (`Ok(None)`; the caller
    // falls back to their existing executor).
    let analysis = plan.capability_analysis(ctx);
    if let Some(projection) = analysis.first_local_rejection_projection() {
        if plan.contains_join() || plan.contains_vector_algorithm() || plan.contains_ml_algorithm() {
            return Ok(Some(CapabilityFirstPipelineAdmission::rejected(None, projection)));
        }
        return Ok(None);
    }
    let envelope_admission = analysis.capability_envelope_admission();
    let pipeline = match compile_pipeline_outcome(plan, &analysis)? {
        PipelineCompilation::Compiled(pipeline) => pipeline,
        // The compiler is the authority on constructability: when it cannot
        // build a pipeline, its structural reason (e.g. a dependency cycle)
        // reports the failure, not a plan-family envelope verdict about a
        // pipeline that does not exist.
        PipelineCompilation::Rejected(projection) => {
            if plan.contains_join() || plan.contains_vector_algorithm() || plan.contains_ml_algorithm() {
                return Ok(Some(CapabilityFirstPipelineAdmission::rejected(None, projection)));
            }
            return Ok(None);
        },
    };
    let converged = !plan.contains_join() && !plan.contains_vector_algorithm() && !plan.contains_ml_algorithm();
    if converged && !pipeline_tasks_within_migrated_contract(&pipeline) {
        // Windows and graph/vector-owned shapes stay on their existing paths
        // until their compiled contracts migrate.
        return Ok(None);
    }
    let vector_task_contract = pipeline
        .tasks()
        .iter()
        .any(|task| matches!(task.kind(), PipelineTaskKind::VectorAlgorithm | PipelineTaskKind::MlAlgorithm))
        && pipeline_tasks_within_migrated_contract(&pipeline);
    let envelope = if pipeline_is_within_migrated_execution_scope(&pipeline) || converged || vector_task_contract {
        analysis.capability_envelope_label()
    } else {
        match envelope_admission {
            CapabilityEnvelopeAdmission::Admitted { envelope } => envelope,
            CapabilityEnvelopeAdmission::Rejected { projection } => {
                return Ok(Some(CapabilityFirstPipelineAdmission::rejected(Some(pipeline), projection)));
            },
        }
    };
    match compiled_bounded_query_plan_support(plan, &analysis, &pipeline)? {
        CompiledBoundedPipelineSupport::Supported => {
            Ok(Some(CapabilityFirstPipelineAdmission::admitted(envelope, pipeline)))
        },
        CompiledBoundedPipelineSupport::Unsupported(projection) => {
            if converged {
                // The generic executor still owns these shapes; a support
                // refusal inside the converged family is a fallback, not a
                // hard refusal of a query that already runs today.
                return Ok(None);
            }
            Ok(Some(CapabilityFirstPipelineAdmission::rejected(Some(pipeline), projection)))
        },
    }
}

/// Execution capabilities the compiled task and resource contracts already
/// decide on their own.
///
/// Join task groups with unary, limit, final-sort, and aggregate stages are
/// admitted by those contracts alone: the named plan-family recognisers keep
/// no veto over such a pipeline and only supply its compatibility label. A
/// resource fanned out to several consumers is covered when it originates in a
/// source read or is a complete immutable frame: frames are refcounted
/// snapshots, so every consumer borrows and releases them through the same
/// path as shared source reads, with no cursor to alias (dynamic-filter
/// application is a no-op for non-source frames, and the contract graph still
/// refuses unresolvable sharing at compile time). Everything else still needs
/// a capability no compiled contract carries yet; windowed composition,
/// shared frame streams, and which join kernels accept another join's
/// retained output; so it keeps the recogniser gate until those contracts
/// exist. Delete this function, and the branch that calls it, once they do.
///
/// Union composition is covered separately: non-join families whose every
/// task sits inside [`pipeline_tasks_within_migrated_contract`] enter the
/// same compiled flow with a union as their first acceptance case.
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
                || resource_originates_in_source_read(pipeline, resource)
        })
}

/// Whether a shared resource's payload originates in a source read, looking
/// through the unary stages that re-stage it.
///
/// A unary stage inherits its input's data form rather than producing one, so a
/// filtered or projected scan is still the scan's stream. Testing only the
/// immediate producer would cover a bare shared scan while rejecting the same
/// scan with a filter above it.
fn resource_originates_in_source_read(pipeline: &CompiledPipeline, resource: &ResidentResource) -> bool {
    let mut producer_id = resource.producer();
    loop {
        let Some(task) = pipeline.tasks().iter().find(|task| task.id() == producer_id) else {
            return false;
        };
        match task.kind() {
            PipelineTaskKind::SourceRead
            | PipelineTaskKind::GpuResidentSource
            | PipelineTaskKind::LiteralBatchSource => return true,
            // `unary_input_form` re-stages a single input, so only that shape
            // can carry a source stream forward.
            PipelineTaskKind::Unary => match task.input_resources() {
                [input] => match pipeline.resources().iter().find(|candidate| candidate.id() == *input) {
                    Some(input) => producer_id = input.producer(),
                    None => return false,
                },
                _ => return false,
            },
            _ => return false,
        }
    }
}
