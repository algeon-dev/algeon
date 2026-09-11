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
        CapabilityFirstCandidateStatus, CapabilityFirstPipelineAdmission, capability_first_bounded_pipeline_admission,
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

/// One capability-first decision for bounded execution and report projection.
///
/// Consumers inspect the admitted pipeline or rejection evidence from this
/// value. Execution consumes it through [`Self::into_execution_admission`] so
/// unsupported compiled-pipeline classifications retain their structured error
/// behavior without re-running capability analysis.
#[derive(Debug)]
pub struct BoundedPipelineDecision {
    admission: Option<BoundedPipelineAdmission>,
    rejection_projection: Option<CapabilityRejectionProjection>,
    execution_error: Option<QueryError>,
}

impl BoundedPipelineDecision {
    /// Admitted analysis, envelope, and compiled pipeline, when applicable.
    pub fn admission(&self) -> Option<&BoundedPipelineAdmission> {
        self.admission.as_ref()
    }

    /// Rejection evidence produced by the same capability decision.
    pub fn rejection_projection(&self) -> Option<&CapabilityRejectionProjection> {
        self.rejection_projection.as_ref()
    }

    /// Converts the decision into the execution-facing admission result.
    pub fn into_execution_admission(self) -> Result<Option<BoundedPipelineAdmission>> {
        if let Some(error) = self.execution_error {
            return Err(error);
        }
        Ok(self.admission)
    }
}

/// Produces the canonical bounded-pipeline decision for a plan and execution mode.
pub fn bounded_pipeline_decision(
    plan: &QueryPlan,
    statistical_aggregate_execution_mode: StatisticalAggregateExecutionMode,
) -> Result<BoundedPipelineDecision> {
    let Some(admission) = capability_first_bounded_pipeline_admission(plan)? else {
        return Ok(BoundedPipelineDecision { admission: None, rejection_projection: None, execution_error: None });
    };
    if !plan.contains_graph_algorithm()
        && let Some(rejection_projection) = row_delete_application_rejection_projection(plan)
    {
        return Ok(BoundedPipelineDecision {
            admission: None,
            rejection_projection: Some(rejection_projection),
            execution_error: None,
        });
    }
    match admission.candidate_status() {
        CapabilityFirstCandidateStatus::Admitted => {
            let mut admission = capability_admission_from_capability_first(admission)?;
            if admission.envelope() == CapabilityEnvelope::ExactStatisticalJoinFinalFullSort
                && !statistical_aggregate_execution_mode.is_datafusion_exact()
            {
                admission = tolerant_statistical_join_final_full_sort_admission(plan, admission)?;
            }
            Ok(BoundedPipelineDecision {
                admission: Some(admission),
                rejection_projection: None,
                execution_error: None,
            })
        },
        CapabilityFirstCandidateStatus::Rejected => {
            let rejection_projection = admission
                .rejection_projection()
                .expect("rejected capability-first admission should carry a projection")
                .clone();
            let execution_error = (plan.contains_graph_algorithm()
                || matches!(
                    rejection_projection.reason,
                    "unsupported_sink_boundary" | "unsupported_compiled_pipeline_envelope"
                ))
            .then(|| compiled_bounded_pipeline_required_error(plan, admission.pipeline(), &rejection_projection));
            Ok(BoundedPipelineDecision {
                admission: None,
                rejection_projection: Some(rejection_projection),
                execution_error,
            })
        },
    }
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
