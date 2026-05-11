use cudf_query_engine::{
    CompiledPipeline, QueryPlan, StrictBoundedContractGateRow, StrictBoundedContractReport,
    capability::{CapabilityAnalysis, ContractStatus},
    compile_pipeline,
};
use datafusion::common::{DataFusionError, Result as DataFusionResult};

use crate::{error::Error, native_exec::NativeCudfExecutionConfig};

pub const STRICT_BOUNDED_NON_JOIN_OUTSIDE_REPORT_SCOPE_REASON: &str =
    "strict_bounded_contract_report_scope_is_compiled_join_pipeline";

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct NativeStrictBoundedContractDiagnostics {
    pub accepted_reports: Vec<StrictBoundedContractReport>,
    pub rejected_blocking_contracts: Vec<StrictBoundedContractGateRow>,
    pub compiled_graphs: Vec<NativeStrictBoundedCompiledGraphDiagnostics>,
    pub non_join_bounded_plan_count: usize,
    pub non_join_bounded_plan_reason: Option<String>,
}

impl NativeStrictBoundedContractDiagnostics {
    pub fn merge(&mut self, other: Self) {
        self.accepted_reports.extend(other.accepted_reports);
        self.rejected_blocking_contracts.extend(other.rejected_blocking_contracts);
        self.compiled_graphs.extend(other.compiled_graphs);
        self.non_join_bounded_plan_count += other.non_join_bounded_plan_count;
        if self.non_join_bounded_plan_reason.is_none() {
            self.non_join_bounded_plan_reason = other.non_join_bounded_plan_reason;
        }
    }

    pub fn is_empty(&self) -> bool {
        self.accepted_reports.is_empty()
            && self.rejected_blocking_contracts.is_empty()
            && self.compiled_graphs.is_empty()
            && self.non_join_bounded_plan_count == 0
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeStrictBoundedCompiledGraphDiagnostics {
    pub task_count: usize,
    pub edge_count: usize,
    pub barrier_count: usize,
    pub resource_count: usize,
    pub gate_count: usize,
    pub sink_kind: String,
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
    execution_config: &NativeCudfExecutionConfig,
) -> DataFusionResult<NativeStrictBoundedContractDiagnostics> {
    if !execution_config.memory_control_policy().is_bounded() {
        return Ok(NativeStrictBoundedContractDiagnostics::default());
    }
    if !plan.contains_join() {
        return Ok(non_join_bounded_diagnostics());
    }

    let analysis = plan.capability_analysis();
    let pipeline = compile_pipeline(plan, &analysis).map_err(strict_bounded_report_error)?;
    let report = StrictBoundedContractReport::from_compiled_pipeline(&analysis, &pipeline)
        .map_err(strict_bounded_report_error)?;
    Ok(NativeStrictBoundedContractDiagnostics {
        accepted_reports: vec![report],
        rejected_blocking_contracts: Vec::new(),
        compiled_graphs: vec![NativeStrictBoundedCompiledGraphDiagnostics::from_pipeline(&pipeline)],
        non_join_bounded_plan_count: 0,
        non_join_bounded_plan_reason: None,
    })
}

pub(super) fn rejected_plan_contract_diagnostics(
    plan: &QueryPlan,
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
        accepted_reports: Vec::new(),
        rejected_blocking_contracts: vec![blocking_row],
        compiled_graphs: Vec::new(),
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
