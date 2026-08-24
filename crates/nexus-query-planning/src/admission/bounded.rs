//! Capability-envelope admission boundary for compiled bounded pipelines.
//!
//! Capability-first bounded-pipeline admission evidence and rejection
//! projection. This is the single bounded admission contract; "capability"
//! appears only in preserved wire names. Runtime execution remains in
//! `exec::bounded`.

use std::collections::BTreeMap;

use super::{
    CompiledBoundedPipelineSupport, compiled_bounded_query_plan_support,
    shared::{
        CapabilityFirstCandidateStatus, CapabilityFirstPipelineAdmission,
        capability_first_bounded_cugraph_pipeline_admission, capability_first_bounded_pipeline_admission,
    },
};
use crate::{
    capability::{
        CapabilityAnalysis, CapabilityEnvelope, CapabilityRejectionCategory, CapabilityRejectionOwner,
        CapabilityRejectionProjection, CapabilityScope,
    },
    error::{QueryError, QueryErrorCode, Result},
    mode::StatisticalAggregateExecutionMode,
    pipeline::{CompiledPipeline, compile_pipeline},
    plan::{PlanNode, QueryPlan, ResultSink},
    source::SourceNode,
};

/// Capability-envelope admission evidence for a compiled bounded pipeline.
///
/// Presence means capability analysis admitted the plan and compilation
/// produced the exact pipeline selected by the bounded controller.
#[derive(Debug, Clone)]
pub struct BoundedPipelineAdmission {
    analysis: CapabilityAnalysis,
    envelope: CapabilityEnvelope,
    pipeline: CompiledPipeline,
}

impl BoundedPipelineAdmission {
    /// Capability facts used to admit the plan.
    pub fn analysis(&self) -> &CapabilityAnalysis {
        &self.analysis
    }

    /// Compiled pipeline admitted for bounded execution.
    pub fn pipeline(&self) -> &CompiledPipeline {
        &self.pipeline
    }

    /// Capability envelope selected by admission.
    pub const fn envelope(&self) -> CapabilityEnvelope {
        self.envelope
    }
}

pub fn bounded_pipeline_admission(plan: &QueryPlan) -> Result<Option<BoundedPipelineAdmission>> {
    let Some(admission) = capability_first_bounded_pipeline_admission(plan)? else {
        return Ok(None);
    };
    if row_delete_application_rejection_projection(plan).is_some() {
        return Ok(None);
    }
    match admission.candidate_status() {
        CapabilityFirstCandidateStatus::Admitted => capability_admission_from_capability_first(admission).map(Some),
        CapabilityFirstCandidateStatus::Rejected => match admission.rejection_reason() {
            Some("unsupported_sink_boundary") | Some("unsupported_compiled_pipeline_envelope") => {
                Err(compiled_bounded_pipeline_required_error(
                    plan,
                    admission.pipeline(),
                    admission
                        .rejection_projection()
                        .expect("rejected capability-first admission should carry a projection"),
                ))
            },
            _ => Ok(None),
        },
    }
}

#[doc(hidden)]
pub fn bounded_cugraph_pipeline_admission(plan: &QueryPlan) -> Result<Option<BoundedPipelineAdmission>> {
    let Some(admission) = capability_first_bounded_cugraph_pipeline_admission(plan)? else {
        return Ok(None);
    };
    match admission.candidate_status() {
        CapabilityFirstCandidateStatus::Admitted => capability_admission_from_capability_first(admission).map(Some),
        CapabilityFirstCandidateStatus::Rejected => Err(compiled_bounded_pipeline_required_error(
            plan,
            admission.pipeline(),
            admission.rejection_projection().expect("rejected cuGraph pipeline admission should carry a projection"),
        )),
    }
}

/// Runs capability-envelope admission for a query plan under the caller-selected
/// floating statistical aggregate execution contract.
///
/// Shape-only admission remains available through
/// [`bounded_pipeline_admission`]. This mode-aware facade is the
/// execution/report boundary: exact statistical final-sort shapes are admitted
/// under either their exact host-Welford contract or a separate GPU-native
/// tolerant contract.
pub fn bounded_pipeline_admission_with_statistical_aggregate_mode(
    plan: &QueryPlan,
    statistical_aggregate_execution_mode: StatisticalAggregateExecutionMode,
) -> Result<Option<BoundedPipelineAdmission>> {
    let Some(admission) = bounded_pipeline_admission(plan)? else {
        return Ok(None);
    };
    if admission.envelope() == CapabilityEnvelope::ExactStatisticalJoinFinalFullSort
        && !statistical_aggregate_execution_mode.is_datafusion_exact()
    {
        return tolerant_statistical_join_final_full_sort_admission(plan, admission).map(Some);
    }
    Ok(Some(admission))
}

/// Returns the shared capability-envelope rejection projection for a join plan.
///
/// This mirrors the one capability-first admission path without converting
/// compiled-runtime ownership failures into execution errors, so report
/// surfaces can project the same rejection evidence.
pub fn bounded_pipeline_rejection_projection(plan: &QueryPlan) -> Result<Option<CapabilityRejectionProjection>> {
    let Some(admission) = capability_first_bounded_pipeline_admission(plan)? else {
        return Ok(None);
    };
    if let Some(projection) = row_delete_application_rejection_projection(plan) {
        return Ok(Some(projection));
    }
    Ok(admission.rejection_projection().cloned())
}

/// Returns the capability-envelope rejection projection after applying the
/// caller-selected floating statistical aggregate execution contract.
pub fn bounded_pipeline_rejection_projection_with_statistical_aggregate_mode(
    plan: &QueryPlan,
    statistical_aggregate_execution_mode: StatisticalAggregateExecutionMode,
) -> Result<Option<CapabilityRejectionProjection>> {
    let Some(admission) = capability_first_bounded_pipeline_admission(plan)? else {
        return Ok(None);
    };
    if let Some(projection) = row_delete_application_rejection_projection(plan) {
        return Ok(Some(projection));
    }
    if admission.envelope() == Some(CapabilityEnvelope::ExactStatisticalJoinFinalFullSort)
        && !statistical_aggregate_execution_mode.is_datafusion_exact()
    {
        return Ok(None);
    }
    Ok(admission.rejection_projection().cloned())
}

fn row_delete_application_rejection_projection(plan: &QueryPlan) -> Option<CapabilityRejectionProjection> {
    let unreachable = plan.unreachable_node_ids();
    plan.nodes().iter().filter(|node| !unreachable.contains(&node.id)).find_map(|node| match &node.kind {
        PlanNode::Source(source) => match source.as_ref() {
            SourceNode::LocalParquet(source) if source.has_row_delete_files() => {
                let position_delete_files =
                    source.files.iter().map(|file| file.position_delete_files.len()).sum::<usize>();
                let equality_delete_files =
                    source.files.iter().map(|file| file.equality_delete_files.len()).sum::<usize>();
                Some(CapabilityRejectionProjection {
                    owner: CapabilityRejectionOwner::Node(node.id),
                    category: CapabilityRejectionCategory::UnsupportedNodeOperator,
                    contract_scope: CapabilityScope::Node,
                    boundary: "row_delete_application",
                    reason: "row_delete_application_unsupported",
                    facts: BTreeMap::from([
                        ("position_delete_files", position_delete_files.to_string()),
                        ("equality_delete_files", equality_delete_files.to_string()),
                    ]),
                })
            },
            _ => None,
        },
        _ => None,
    })
}

fn tolerant_statistical_join_final_full_sort_admission(
    plan: &QueryPlan,
    admission: BoundedPipelineAdmission,
) -> Result<BoundedPipelineAdmission> {
    let analysis = admission.analysis;
    let pipeline = compile_pipeline(plan, &analysis)?;
    match compiled_bounded_query_plan_support(plan, &pipeline)? {
        CompiledBoundedPipelineSupport::Supported => Ok(BoundedPipelineAdmission {
            analysis,
            envelope: CapabilityEnvelope::TolerantStatisticalJoinFinalFullSort,
            pipeline,
        }),
        CompiledBoundedPipelineSupport::Unsupported => Err(QueryError::new(
            QueryErrorCode::UnsupportedPlanShape,
            "nexus_query_engine.bounded.tolerant_statistical_admission",
            "tolerant statistical join final full sort requires compiled bounded pipeline support",
        )
        .with_fact("capability_envelope", CapabilityEnvelope::TolerantStatisticalJoinFinalFullSort.as_str())
        .with_fact("compiled_task_count", pipeline.tasks().len())
        .with_fact("compiled_edge_count", pipeline.edges().len())),
    }
}

fn capability_admission_from_capability_first(
    admission: CapabilityFirstPipelineAdmission,
) -> Result<BoundedPipelineAdmission> {
    let envelope = admission.envelope().ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.bounded.capability_admission",
            "admitted capability candidate must include a selected envelope",
        )
        .with_fact("execution_mode", crate::mode::CapabilityEnvelopeStatus::Admitted.as_str())
        .with_fact("candidate_status", "admitted")
    })?;
    let (analysis, pipeline) = admission.into_parts();
    let pipeline = pipeline.ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.bounded.capability_admission",
            "admitted capability candidate must include a compiled pipeline",
        )
        .with_fact("execution_mode", crate::mode::CapabilityEnvelopeStatus::Admitted.as_str())
        .with_fact("candidate_status", "admitted")
    })?;
    Ok(BoundedPipelineAdmission { analysis, envelope, pipeline })
}

fn compiled_bounded_pipeline_required_error(
    plan: &QueryPlan,
    pipeline: Option<&crate::pipeline::CompiledPipeline>,
    projection: &CapabilityRejectionProjection,
) -> QueryError {
    let sink_kind = match plan.sink() {
        ResultSink::GpuResident { .. } => "gpu_resident",
        ResultSink::HostArrow { .. } => "host_arrow",
    };
    let mut error = QueryError::new(
        QueryErrorCode::UnsupportedPlanShape,
        "nexus_query_engine.bounded.compiled_pipeline_required",
        "capability-admitted bounded join execution requires the compiled bounded pipeline runtime",
    )
    .with_fact("compiled_pipeline_required", "true")
    .with_fact("compiled_pipeline_rejection_reason", projection.reason)
    .with_fact("result_sink_kind", sink_kind);

    if let Some(pipeline) = pipeline {
        error = error
            .with_fact("compiled_task_count", pipeline.tasks().len())
            .with_fact("compiled_edge_count", pipeline.edges().len())
            .with_fact("compiled_barrier_count", pipeline.barriers().len())
            .with_fact("compiled_resource_count", pipeline.resources().len())
            .with_fact("compiled_sink_kind", pipeline.sink().kind().as_str());
    }

    // Projection facts may repeat pipeline-derived keys; the pipeline values
    // above are authoritative, so only absent keys are added here.
    for (key, value) in &projection.facts {
        error = error.with_fact_if_absent(key, value.as_str());
    }

    error
}
