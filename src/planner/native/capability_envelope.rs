use datafusion::common::{DataFusionError, Result as DataFusionResult};
use nexus_query_engine::{
    admission::{BoundedPipelineDecision, bounded_pipeline_decision},
    contract_report::{CapabilityContractReport, CapabilityDecisionRow},
    mode::CapabilityEnvelopeStatus,
    pipeline::CompiledPipeline,
    plan::{PlanNode, QueryPlan},
    source::SourceNode,
};

use crate::{error::Error, native_exec::NexusNativeExecutionConfig, report::GpuCandidate};

/// Stable compatibility reason used when report rows omit bounded non-join plans.
pub const CAPABILITY_NON_JOIN_OUTSIDE_REPORT_SCOPE_REASON: &str =
    "capability_contract_report_scope_is_compiled_join_pipeline";

/// Adapter-side capability-contract diagnostics collected during planning.
///
/// This is a projection of native-engine contract reports for DataFusion report
/// rows. It does not perform admission or pipeline scheduling itself.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct NativeCapabilityContractDiagnostics {
    /// Candidates accepted by the optimizer with capability reports attached.
    pub accepted_candidates: Vec<NativeCapabilityAcceptedCandidateDiagnostics>,
    /// Structurally supported candidates waiting for resident producer facts.
    pub deferred_candidates: Vec<NativeCapabilityDeferredCandidateDiagnostics>,
    /// Candidates rejected before execution with their first blocking decision.
    pub rejected_candidates: Vec<NativeCapabilityRejectedCandidateDiagnostics>,
    /// Bounded non-join plans seen while compatibility reports cover compiled joins.
    pub non_join_bounded_plan_count: usize,
    /// Stable reason explaining why non-join bounded plans lack gate rows.
    pub non_join_bounded_plan_reason: Option<String>,
}

impl NativeCapabilityContractDiagnostics {
    /// Appends accepted/rejected candidate evidence from another optimizer stage.
    pub fn merge(&mut self, other: Self) {
        self.accepted_candidates.extend(other.accepted_candidates);
        self.deferred_candidates.extend(other.deferred_candidates);
        self.rejected_candidates.extend(other.rejected_candidates);
        self.non_join_bounded_plan_count += other.non_join_bounded_plan_count;
        if self.non_join_bounded_plan_reason.is_none() {
            self.non_join_bounded_plan_reason = other.non_join_bounded_plan_reason;
        }
    }

    /// Returns true when no capability-contract evidence was captured.
    pub fn is_empty(&self) -> bool {
        self.accepted_candidates.is_empty()
            && self.deferred_candidates.is_empty()
            && self.rejected_candidates.is_empty()
            && self.non_join_bounded_plan_count == 0
    }
}

/// Planner decision for a candidate whose composed evidence awaits producer binding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeCapabilityDeferredCandidateDiagnostics {
    /// Optimizer candidate that reached the deferred capability stage.
    pub candidate: GpuCandidate,
    /// Stable stage that emitted this decision.
    pub planning_stage: String,
    /// Report carrying the deferred status without a parallel event store.
    pub report: CapabilityContractReport,
}

/// Capability report and compiled graph facts for an accepted candidate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeCapabilityAcceptedCandidateDiagnostics {
    /// Optimizer candidate that produced these diagnostics.
    pub candidate: GpuCandidate,
    /// Capability report.
    pub report: CapabilityContractReport,
    /// Graph-size diagnostics for the compiled pipeline.
    pub compiled_graph: NativeCapabilityCompiledGraphDiagnostics,
}

/// First capability decision that blocked a rejected candidate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeCapabilityRejectedCandidateDiagnostics {
    /// Optimizer candidate that was rejected.
    pub candidate: GpuCandidate,
    /// First rejected row from capability analysis.
    pub blocking_contract: CapabilityDecisionRow,
}

/// Report-facing graph summary for a compiled capability envelope.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeCapabilityCompiledGraphDiagnostics {
    /// Number of compiled tasks.
    pub task_count: usize,
    /// Number of compiled dependency edges.
    pub edge_count: usize,
    /// Number of compiled barriers.
    pub barrier_count: usize,
    /// Number of resident resources tracked by the graph.
    pub resource_count: usize,
    /// Final sink boundary kind.
    pub sink_kind: String,
}

impl NativeCapabilityCompiledGraphDiagnostics {
    fn from_pipeline(pipeline: &CompiledPipeline) -> Self {
        Self {
            task_count: pipeline.tasks().len(),
            edge_count: pipeline.edges().len(),
            barrier_count: pipeline.barriers().len(),
            resource_count: pipeline.resources().len(),
            sink_kind: pipeline.sink().kind().as_str().to_string(),
        }
    }
}

/// Decides the executable bounded plan once for the candidate flow.
///
/// Returns `None` when there is nothing to decide yet: the policy is
/// unbounded, the plan has no join, or a composed consumer still awaits its
/// producer binding. Callers pass the same decision to the boundary check and
/// to the accepted/rejected diagnostics instead of re-deriving it.
pub(super) fn native_bounded_decision(
    plan: &QueryPlan,
    execution_config: &NexusNativeExecutionConfig,
) -> DataFusionResult<Option<BoundedPipelineDecision>> {
    if !execution_config.memory_control_policy().is_bounded() || !plan.contains_join() {
        return Ok(None);
    }
    if composed_admission_requires_producer_binding(plan) {
        return Ok(None);
    }
    bounded_pipeline_decision(plan, execution_config.statistical_aggregate_execution_mode())
        .map(Some)
        .map_err(|error| capability_contract_report_error(stamp_native_error_correlation(error, execution_config)))
}

pub(super) fn accepted_plan_contract_diagnostics(
    plan: &QueryPlan,
    candidate: GpuCandidate,
    execution_config: &NexusNativeExecutionConfig,
    bounded_decision: Option<&BoundedPipelineDecision>,
) -> DataFusionResult<NativeCapabilityContractDiagnostics> {
    if !execution_config.memory_control_policy().is_bounded() {
        return Ok(NativeCapabilityContractDiagnostics::default());
    }
    if !plan.contains_join() {
        return Ok(non_join_capability_diagnostics());
    }
    if composed_admission_requires_producer_binding(plan) {
        let decision = CapabilityDecisionRow::deferred_composed_producer_binding(0).map_err(|error| {
            capability_contract_report_error(stamp_native_error_correlation(error, execution_config))
        })?;
        let report = CapabilityContractReport::from_decision_rows(CapabilityEnvelopeStatus::Deferred, vec![decision])
            .map_err(|error| {
            capability_contract_report_error(stamp_native_error_correlation(error, execution_config))
        })?;
        return Ok(NativeCapabilityContractDiagnostics {
            deferred_candidates: vec![NativeCapabilityDeferredCandidateDiagnostics {
                candidate,
                planning_stage: "physical_optimizer".to_string(),
                report: stamp_capability_report_correlation(report, execution_config),
            }],
            ..Default::default()
        });
    }

    let Some(decision) = bounded_decision else {
        return Err(capability_contract_candidate_error(
            "bounded rewrite has no capability-envelope admission evidence",
        ));
    };
    let admission = decision.admission().ok_or_else(|| {
        capability_contract_candidate_error("bounded rewrite has no capability-envelope admission evidence")
    })?;
    let report = stamp_capability_report_correlation(
        CapabilityContractReport::from_admission(admission).map_err(|error| {
            capability_contract_report_error(stamp_native_error_correlation(error, execution_config))
        })?,
        execution_config,
    );
    Ok(NativeCapabilityContractDiagnostics {
        accepted_candidates: vec![NativeCapabilityAcceptedCandidateDiagnostics {
            candidate,
            report,
            compiled_graph: NativeCapabilityCompiledGraphDiagnostics::from_pipeline(admission.pipeline()),
        }],
        deferred_candidates: Vec::new(),
        rejected_candidates: Vec::new(),
        non_join_bounded_plan_count: 0,
        non_join_bounded_plan_reason: None,
    })
}

/// Reads the boundary verdict off an already-made decision.
///
/// A missing decision means there was nothing to decide (unbounded policy,
/// joinless plan, or a composed consumer awaiting producer binding) and never
/// rejects.
pub(super) fn native_execution_boundary_is_rejected_on_decision(
    bounded_decision: Option<&BoundedPipelineDecision>,
) -> bool {
    matches!(bounded_decision, Some(decision) if decision.admission().is_none())
}

/// A plan with a GPU-resident source is a composed consumer whose producer
/// output is not yet available at this planning stage: the producer prepares
/// and executes separately, after this physical-optimizer pass runs.
fn composed_admission_requires_producer_binding(plan: &QueryPlan) -> bool {
    plan.nodes().iter().any(
        |node| matches!(&node.kind, PlanNode::Source(source) if matches!(source.as_ref(), SourceNode::GpuResident(_))),
    )
}

pub(super) fn rejected_plan_contract_diagnostics(
    plan: &QueryPlan,
    candidate: GpuCandidate,
    execution_config: &NexusNativeExecutionConfig,
    bounded_decision: Option<&BoundedPipelineDecision>,
) -> DataFusionResult<NativeCapabilityContractDiagnostics> {
    if !execution_config.memory_control_policy().is_bounded() || !plan.contains_join() {
        return Ok(NativeCapabilityContractDiagnostics::default());
    }

    let Some(blocking_row) = first_blocking_contract_row(plan, execution_config, bounded_decision)? else {
        return Ok(NativeCapabilityContractDiagnostics::default());
    };
    Ok(NativeCapabilityContractDiagnostics {
        accepted_candidates: Vec::new(),
        deferred_candidates: Vec::new(),
        rejected_candidates: vec![NativeCapabilityRejectedCandidateDiagnostics {
            candidate,
            blocking_contract: blocking_row,
        }],
        non_join_bounded_plan_count: 0,
        non_join_bounded_plan_reason: None,
    })
}

fn first_blocking_contract_row(
    plan: &QueryPlan,
    execution_config: &NexusNativeExecutionConfig,
    bounded_decision: Option<&BoundedPipelineDecision>,
) -> DataFusionResult<Option<CapabilityDecisionRow>> {
    if let Some(projection) = bounded_decision.and_then(|decision| decision.rejection_projection()) {
        return CapabilityDecisionRow::from_rejection_projection(0, projection).map(Some).map_err(|error| {
            capability_contract_report_error(stamp_native_error_correlation(error, execution_config))
        });
    }

    let analysis = plan.capability_analysis();
    let report = CapabilityContractReport::from_capability_analysis(&analysis)
        .map_err(|error| capability_contract_report_error(stamp_native_error_correlation(error, execution_config)))?;
    Ok(report.decision_rows().first().cloned())
}

fn non_join_capability_diagnostics() -> NativeCapabilityContractDiagnostics {
    NativeCapabilityContractDiagnostics {
        non_join_bounded_plan_count: 1,
        non_join_bounded_plan_reason: Some(CAPABILITY_NON_JOIN_OUTSIDE_REPORT_SCOPE_REASON.to_string()),
        ..Default::default()
    }
}

fn capability_contract_report_error(error: nexus_query_engine::error::QueryError) -> DataFusionError {
    Error::from_native(
        "datafusion_nexus.native.capability_contract_diagnostics",
        "capability-contract diagnostics could not be built",
        error,
    )
    .into()
}

fn capability_contract_candidate_error(message: &'static str) -> DataFusionError {
    Error::new(
        crate::error::ErrorCode::InvalidExecutionPlan,
        "datafusion_nexus.native.capability_contract_diagnostics",
        message,
    )
    .into()
}

fn stamp_capability_report_correlation(
    mut report: CapabilityContractReport,
    execution_config: &NexusNativeExecutionConfig,
) -> CapabilityContractReport {
    if let Some(query_id) = execution_config.query_id() {
        report = report.with_query_id(query_id.clone());
    }
    if let Some(trace_id) = execution_config.trace_id() {
        report = report.with_trace_id(trace_id.clone());
    }
    report
}

fn stamp_native_error_correlation(
    mut error: nexus_query_engine::error::QueryError,
    execution_config: &NexusNativeExecutionConfig,
) -> nexus_query_engine::error::QueryError {
    if let Some(query_id) = execution_config.query_id() {
        error = error.with_query_id(query_id.clone());
    }
    if let Some(trace_id) = execution_config.trace_id() {
        error = error.with_trace_id(trace_id.clone());
    }
    error
}
