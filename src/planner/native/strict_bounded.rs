use cudf_query_engine::{
    CompiledPipeline, QueryPlan, StrictBoundedContractGateRow, StrictBoundedContractReport,
    capability::{CapabilityAnalysis, ContractStatus},
    strict_bounded_pipeline_admission,
};
use datafusion::common::{DataFusionError, Result as DataFusionResult};

use crate::{error::Error, native_exec::NativeCudfExecutionConfig, report::GpuCandidate};

/// Stable reason used when strict report rows intentionally omit bounded non-join plans.
pub const STRICT_BOUNDED_NON_JOIN_OUTSIDE_REPORT_SCOPE_REASON: &str =
    "strict_bounded_contract_report_scope_is_compiled_join_pipeline";

/// Adapter-side strict bounded contract diagnostics collected during planning.
///
/// This is a projection of native-engine contract reports for DataFusion report
/// rows. It does not perform admission or pipeline scheduling itself.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct NativeStrictBoundedContractDiagnostics {
    /// Candidates accepted by the optimizer with strict gate rows attached.
    pub accepted_candidates: Vec<NativeStrictBoundedAcceptedCandidateDiagnostics>,
    /// Candidates rejected before execution with their first blocking gate.
    pub rejected_candidates: Vec<NativeStrictBoundedRejectedCandidateDiagnostics>,
    /// Bounded non-join plans seen while strict reports cover compiled joins.
    pub non_join_bounded_plan_count: usize,
    /// Stable reason explaining why non-join bounded plans lack gate rows.
    pub non_join_bounded_plan_reason: Option<String>,
}

impl NativeStrictBoundedContractDiagnostics {
    /// Appends accepted/rejected candidate evidence from another optimizer phase.
    pub fn merge(&mut self, other: Self) {
        self.accepted_candidates.extend(other.accepted_candidates);
        self.rejected_candidates.extend(other.rejected_candidates);
        self.non_join_bounded_plan_count += other.non_join_bounded_plan_count;
        if self.non_join_bounded_plan_reason.is_none() {
            self.non_join_bounded_plan_reason = other.non_join_bounded_plan_reason;
        }
    }

    /// Returns true when no strict bounded report evidence was captured.
    pub fn is_empty(&self) -> bool {
        self.accepted_candidates.is_empty()
            && self.rejected_candidates.is_empty()
            && self.non_join_bounded_plan_count == 0
    }
}

/// Strict bounded report and compiled graph facts for an accepted candidate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeStrictBoundedAcceptedCandidateDiagnostics {
    /// Optimizer candidate that produced these diagnostics.
    pub candidate: GpuCandidate,
    /// Gate-first strict bounded contract report.
    pub report: StrictBoundedContractReport,
    /// Graph-size diagnostics for the compiled pipeline.
    pub compiled_graph: NativeStrictBoundedCompiledGraphDiagnostics,
}

/// First strict bounded gate that blocked a rejected candidate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeStrictBoundedRejectedCandidateDiagnostics {
    /// Optimizer candidate that was rejected.
    pub candidate: GpuCandidate,
    /// First non-modeled contract row from capability analysis.
    pub blocking_contract: StrictBoundedContractGateRow,
}

/// Report-facing graph summary for a compiled strict bounded pipeline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeStrictBoundedCompiledGraphDiagnostics {
    /// Number of compiled tasks.
    pub task_count: usize,
    /// Number of compiled dependency edges.
    pub edge_count: usize,
    /// Number of compiled barriers.
    pub barrier_count: usize,
    /// Number of resident resources tracked by the graph.
    pub resource_count: usize,
    /// Number of reservation gates attached to the graph.
    pub gate_count: usize,
    /// Final sink boundary kind.
    pub sink_kind: String,
    /// Reservation gates owned by the sink boundary.
    pub sink_gate_count: usize,
}

impl NativeStrictBoundedCompiledGraphDiagnostics {
    fn from_pipeline(pipeline: &CompiledPipeline) -> Self {
        Self {
            task_count: pipeline.tasks().len(),
            edge_count: pipeline.edges().len(),
            barrier_count: pipeline.barriers().len(),
            resource_count: pipeline.resources().len(),
            gate_count: pipeline.gates().len(),
            sink_kind: pipeline.sink().kind().as_str().to_string(),
            sink_gate_count: pipeline.sink().gates().len(),
        }
    }
}

pub(super) fn accepted_plan_contract_diagnostics(
    plan: &QueryPlan,
    candidate: GpuCandidate,
    execution_config: &NativeCudfExecutionConfig,
) -> DataFusionResult<NativeStrictBoundedContractDiagnostics> {
    if !execution_config.memory_control_policy().is_bounded() {
        return Ok(NativeStrictBoundedContractDiagnostics::default());
    }
    if !plan.contains_join() {
        return Ok(non_join_bounded_diagnostics());
    }

    let admission = strict_bounded_pipeline_admission(plan)
        .map_err(strict_bounded_report_error)?
        .ok_or_else(|| strict_bounded_candidate_error("strict bounded rewrite has no strict admission evidence"))?;
    let report = StrictBoundedContractReport::from_compiled_pipeline(admission.analysis(), admission.pipeline())
        .map_err(strict_bounded_report_error)?;
    Ok(NativeStrictBoundedContractDiagnostics {
        accepted_candidates: vec![NativeStrictBoundedAcceptedCandidateDiagnostics {
            candidate,
            report,
            compiled_graph: NativeStrictBoundedCompiledGraphDiagnostics::from_pipeline(admission.pipeline()),
        }],
        rejected_candidates: Vec::new(),
        non_join_bounded_plan_count: 0,
        non_join_bounded_plan_reason: None,
    })
}

pub(super) fn strict_bounded_admission_is_rejected(plan: &QueryPlan) -> DataFusionResult<bool> {
    match strict_bounded_pipeline_admission(plan) {
        Ok(Some(_)) => Ok(false),
        Ok(None) => Ok(true),
        Err(error) if error.code() == cudf_query_engine::QueryErrorCode::UnsupportedPlanShape => Ok(true),
        Err(error) => Err(strict_bounded_report_error(error)),
    }
}

pub(super) fn rejected_plan_contract_diagnostics(
    plan: &QueryPlan,
    candidate: GpuCandidate,
    execution_config: &NativeCudfExecutionConfig,
) -> DataFusionResult<NativeStrictBoundedContractDiagnostics> {
    if !execution_config.memory_control_policy().is_bounded() || !plan.contains_join() {
        return Ok(NativeStrictBoundedContractDiagnostics::default());
    }

    let analysis = plan.capability_analysis();
    let Some(blocking_row) = first_blocking_contract_row(&analysis)? else {
        return Ok(NativeStrictBoundedContractDiagnostics::default());
    };
    Ok(NativeStrictBoundedContractDiagnostics {
        accepted_candidates: Vec::new(),
        rejected_candidates: vec![NativeStrictBoundedRejectedCandidateDiagnostics {
            candidate,
            blocking_contract: blocking_row,
        }],
        non_join_bounded_plan_count: 0,
        non_join_bounded_plan_reason: None,
    })
}

fn first_blocking_contract_row(
    analysis: &CapabilityAnalysis,
) -> DataFusionResult<Option<StrictBoundedContractGateRow>> {
    let report =
        StrictBoundedContractReport::from_capability_analysis(analysis).map_err(strict_bounded_report_error)?;
    Ok(report.gate_rows().iter().find(|row| row.contract_status != ContractStatus::Modeled).cloned())
}

fn non_join_bounded_diagnostics() -> NativeStrictBoundedContractDiagnostics {
    NativeStrictBoundedContractDiagnostics {
        non_join_bounded_plan_count: 1,
        non_join_bounded_plan_reason: Some(STRICT_BOUNDED_NON_JOIN_OUTSIDE_REPORT_SCOPE_REASON.to_string()),
        ..Default::default()
    }
}

fn strict_bounded_report_error(error: cudf_query_engine::QueryError) -> DataFusionError {
    Error::from_native(
        "datafusion_cudf.native.strict_bounded_contract_diagnostics",
        "strict bounded contract diagnostics could not be built",
        error,
    )
    .into()
}

fn strict_bounded_candidate_error(message: &'static str) -> DataFusionError {
    Error::new(
        crate::error::ErrorCode::InvalidExecutionPlan,
        "datafusion_cudf.native.strict_bounded_contract_diagnostics",
        message,
    )
    .into()
}
