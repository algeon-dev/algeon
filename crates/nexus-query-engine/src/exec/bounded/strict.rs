//! Strict bounded-pipeline execution boundary.
//!
//! This module is the strict-only facade for compiled bounded execution. Public
//! and crate-local strict exports should pass through this boundary instead of
//! importing non-strict admission or report surfaces.

pub(in crate::exec::bounded) use super::compiled::join_strategy::CompiledJoinExecutionStrategy;
#[doc(hidden)]
pub use super::compiled::{
    CompiledBoundedPipelineSupport, CompiledGpuResource, CompiledGpuTaskExecutor, CompiledPipelineScheduler,
    CompiledPipelineState, CompiledTaskAdmission, CompiledTaskExecutor, CompiledTaskOutput, ReleasedCompiledResource,
    ResidentFrameStreamProducerTestHarness, YieldedFrameCounterTestHarness, compiled_bounded_pipeline_support,
    compiled_bounded_query_plan_support,
};
use super::{
    admission::{
        CapabilityFirstCandidateStatus, CapabilityFirstPipelineAdmission, capability_first_bounded_pipeline_admission,
    },
    compiled::execute_compiled_pipeline,
};
use crate::{
    capability::{
        CapabilityAnalysis, CapabilityRejectionProjection, ContractQuality, ContractStatus, ReservationRequirement,
        StrictBoundedEnvelope, SubgraphInvariantKind,
    },
    error::{QueryError, QueryErrorCode, Result},
    exec::types::{GpuResidentInputMap, QueryOutput},
    metrics::QueryMetrics,
    mode::{NativePipelineExecutionMode, StatisticalAggregateExecutionMode},
    pipeline::{CompiledPipeline, compile_pipeline},
    plan::{QueryPlan, ResultSink},
    runtime::{ExecutionContext, ReservationKind},
};

/// Strict bounded admission evidence for a compiled bounded pipeline.
///
/// Presence means capability analysis admitted the plan and compilation
/// produced the exact pipeline that strict execution will run.
#[derive(Debug, Clone)]
pub struct StrictBoundedPipelineAdmission {
    analysis: CapabilityAnalysis,
    envelope: StrictBoundedEnvelope,
    pipeline: CompiledPipeline,
}

impl StrictBoundedPipelineAdmission {
    /// Capability facts used to admit the plan.
    pub fn analysis(&self) -> &CapabilityAnalysis {
        &self.analysis
    }

    /// Compiled pipeline admitted for strict bounded execution.
    pub fn pipeline(&self) -> &CompiledPipeline {
        &self.pipeline
    }

    /// Strict bounded envelope selected by capability admission.
    pub const fn envelope(&self) -> StrictBoundedEnvelope {
        self.envelope
    }

    /// Native pipeline mode represented by this admission.
    pub const fn native_mode(&self) -> NativePipelineExecutionMode {
        NativePipelineExecutionMode::StrictBounded
    }
}

/// Run strict bounded admission for a query plan.
///
/// `Ok(None)` means the plan is outside strict compiled-pipeline support.
/// `Ok(Some(_))` means every strict gate required for compilation was modeled.
pub fn strict_bounded_pipeline_admission(plan: &QueryPlan) -> Result<Option<StrictBoundedPipelineAdmission>> {
    let Some(admission) = capability_first_bounded_pipeline_admission(plan)? else {
        return Ok(None);
    };
    match admission.candidate_status() {
        CapabilityFirstCandidateStatus::Admitted => strict_admission_from_capability_first(admission).map(Some),
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

/// Run strict bounded admission for a query plan under the caller-selected
/// floating statistical aggregate execution contract.
///
/// Shape-only admission remains available through
/// [`strict_bounded_pipeline_admission`]. This mode-aware facade is the
/// execution/report boundary: exact statistical final-sort shapes are admitted
/// under either their exact host-Welford contract or a separate GPU-native
/// tolerant contract.
pub fn strict_bounded_pipeline_admission_with_statistical_aggregate_mode(
    plan: &QueryPlan,
    statistical_aggregate_execution_mode: StatisticalAggregateExecutionMode,
) -> Result<Option<StrictBoundedPipelineAdmission>> {
    let Some(admission) = strict_bounded_pipeline_admission(plan)? else {
        return Ok(None);
    };
    if admission.envelope() == StrictBoundedEnvelope::ExactStatisticalJoinFinalFullSort
        && !statistical_aggregate_execution_mode.is_datafusion_exact()
    {
        return tolerant_statistical_join_final_full_sort_admission(plan, admission).map(Some);
    }
    Ok(Some(admission))
}

/// Returns the shared strict bounded rejection projection for a join plan.
///
/// This mirrors the strict/adaptive capability-first admission path without
/// converting compiled-runtime ownership failures into execution errors, so
/// report surfaces can project the same rejection evidence.
pub fn strict_bounded_pipeline_rejection_projection(plan: &QueryPlan) -> Result<Option<CapabilityRejectionProjection>> {
    let Some(admission) = capability_first_bounded_pipeline_admission(plan)? else {
        return Ok(None);
    };
    Ok(admission.rejection_projection().cloned())
}

/// Returns the strict bounded rejection projection after applying the
/// caller-selected floating statistical aggregate execution contract.
pub fn strict_bounded_pipeline_rejection_projection_with_statistical_aggregate_mode(
    plan: &QueryPlan,
    statistical_aggregate_execution_mode: StatisticalAggregateExecutionMode,
) -> Result<Option<CapabilityRejectionProjection>> {
    let Some(admission) = capability_first_bounded_pipeline_admission(plan)? else {
        return Ok(None);
    };
    if admission.envelope() == Some(StrictBoundedEnvelope::ExactStatisticalJoinFinalFullSort)
        && !statistical_aggregate_execution_mode.is_datafusion_exact()
    {
        return Ok(None);
    }
    Ok(admission.rejection_projection().cloned())
}

fn tolerant_statistical_join_final_full_sort_admission(
    plan: &QueryPlan,
    admission: StrictBoundedPipelineAdmission,
) -> Result<StrictBoundedPipelineAdmission> {
    let mut analysis = admission.analysis;
    rewrite_exact_statistical_join_final_sort_requirements_for_tolerant_mode(&mut analysis);
    let pipeline = compile_pipeline(plan, &analysis)?;
    match compiled_bounded_query_plan_support(plan, &pipeline)? {
        CompiledBoundedPipelineSupport::Supported => Ok(StrictBoundedPipelineAdmission {
            analysis,
            envelope: StrictBoundedEnvelope::TolerantStatisticalJoinFinalFullSort,
            pipeline,
        }),
        CompiledBoundedPipelineSupport::Unsupported => Err(QueryError::new(
            QueryErrorCode::UnsupportedPlanShape,
            "nexus_query_engine.bounded.strict_tolerant_statistical_admission",
            "tolerant statistical join final full sort requires compiled bounded pipeline support",
        )
        .with_fact("strict_bounded_envelope", StrictBoundedEnvelope::TolerantStatisticalJoinFinalFullSort.as_str())
        .with_fact("compiled_task_count", pipeline.tasks().len())
        .with_fact("compiled_edge_count", pipeline.edges().len())),
    }
}

fn rewrite_exact_statistical_join_final_sort_requirements_for_tolerant_mode(analysis: &mut CapabilityAnalysis) {
    for invariant in &mut analysis.invariants {
        if invariant.kind != SubgraphInvariantKind::StatisticalJoinFinalSortExactness {
            continue;
        }
        invariant
            .reservation_requirements
            .retain(|requirement| requirement.required_contract == "final_full_sort_output_cardinality_bound");
        invariant.reservation_requirements.push(ReservationRequirement::modeled(
            ReservationKind::CapabilityContract,
            StrictBoundedEnvelope::TolerantStatisticalJoinFinalFullSort.as_str(),
            "gpu_native_tolerant_statistical_aggregate_contract",
            ContractQuality::CapabilityProof,
        ));
    }

    analysis.requirements = analysis
        .nodes
        .iter()
        .flat_map(|node| node.local_requirements.iter())
        .chain(analysis.edges.iter().flat_map(|edge| edge.reservation_requirements.iter()))
        .chain(analysis.invariants.iter().flat_map(|invariant| invariant.reservation_requirements.iter()))
        .cloned()
        .collect();
    debug_assert!(
        analysis.requirements.iter().all(|requirement| {
            !matches!(
                (requirement.status, requirement.required_contract),
                (
                    ContractStatus::Modeled,
                    "host_welford_input_materialization_preflight" | "host_welford_output_upload_reservation"
                )
            )
        }),
        "tolerant statistical contract must not carry exact host-Welford gates"
    );
}

fn strict_admission_from_capability_first(
    admission: CapabilityFirstPipelineAdmission,
) -> Result<StrictBoundedPipelineAdmission> {
    let envelope = admission.envelope().ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.bounded.strict_admission",
            "admitted strict bounded candidate must include a selected envelope",
        )
        .with_fact("execution_mode", NativePipelineExecutionMode::StrictBounded.as_str())
        .with_fact("candidate_status", "admitted")
    })?;
    let (analysis, pipeline) = admission.into_parts();
    let pipeline = pipeline.ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.bounded.strict_admission",
            "admitted strict bounded candidate must include a compiled pipeline",
        )
        .with_fact("execution_mode", NativePipelineExecutionMode::StrictBounded.as_str())
        .with_fact("candidate_status", "admitted")
    })?;
    Ok(StrictBoundedPipelineAdmission { analysis, envelope, pipeline })
}

pub(in crate::exec) fn try_execute_strict_bounded_pipeline(
    context: &ExecutionContext,
    plan: &QueryPlan,
    gpu_inputs: &GpuResidentInputMap,
    metrics: &mut QueryMetrics,
) -> Result<Option<QueryOutput>> {
    let Some(admission) = strict_bounded_pipeline_admission_with_statistical_aggregate_mode(
        plan,
        context.statistical_aggregate_execution_mode(),
    )?
    else {
        return Ok(None);
    };
    debug_assert_eq!(admission.native_mode(), NativePipelineExecutionMode::StrictBounded);
    execute_compiled_pipeline(context, plan, admission.pipeline(), gpu_inputs, metrics).map(Some)
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
    .with_fact("legacy_bounded_join_path_bridge", "disabled")
    .with_fact("result_sink_kind", sink_kind);

    if let Some(pipeline) = pipeline {
        error = error
            .with_fact("compiled_task_count", pipeline.tasks().len())
            .with_fact("compiled_edge_count", pipeline.edges().len())
            .with_fact("compiled_barrier_count", pipeline.barriers().len())
            .with_fact("compiled_resource_count", pipeline.resources().len())
            .with_fact("compiled_gate_count", pipeline.gates().len())
            .with_fact("compiled_sink_kind", pipeline.sink().kind().as_str());
    }

    // Projection facts may repeat pipeline-derived keys; the pipeline values
    // above are authoritative, so only absent keys are added here.
    for (key, value) in &projection.facts {
        error = error.with_fact_if_absent(key, value.as_str());
    }

    error
}
