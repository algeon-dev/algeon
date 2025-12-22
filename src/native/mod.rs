mod literal;
mod lowering;
mod normalize;
mod pruning;
mod source;
mod validate;

use std::sync::Arc;

use cudf_query_engine::{
    PlanNode as NativePlanNode, QueryPlan as NativeQueryPlan, SourceDiagnostics as NativeSourceDiagnostics,
    SourceNode as NativeSourceNode,
};
use datafusion::physical_plan::ExecutionPlan;
use lowering::lower_normalized_plan;
use normalize::normalize_native_plan;
use source::source_diagnostics_for_normalized_plan;
use validate::native_zero_column_count_source_is_safe;

use crate::report::PlanningReport;

#[derive(Debug, Clone)]
pub struct NativeLoweringResult {
    pub plans: Vec<NativeQueryPlan>,
    pub report: PlanningReport,
    pub source_diagnostics: Vec<NativeSourceDiagnostics>,
}

impl NativeLoweringResult {
    pub fn candidate_count(&self) -> usize {
        self.report.outcomes().len()
    }

    pub fn rewritten_count(&self) -> usize {
        self.report.rewritten_count()
    }

    pub fn rejected_count(&self) -> usize {
        self.report.rejected_count()
    }
}

pub fn lower_physical_plan_to_native(plan: Arc<dyn ExecutionPlan>) -> NativeLoweringResult {
    lower_physical_plan_to_native_with_mode(plan, NativeLoweringMode::Final)
}

pub(crate) fn lower_count_only_physical_plan_to_native(plan: Arc<dyn ExecutionPlan>) -> NativeLoweringResult {
    lower_physical_plan_to_native_with_mode(plan, NativeLoweringMode::CountOnly)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NativeLoweringMode {
    Final,
    CountOnly,
}

fn lower_physical_plan_to_native_with_mode(
    plan: Arc<dyn ExecutionPlan>,
    mode: NativeLoweringMode,
) -> NativeLoweringResult {
    match normalize_native_plan(plan.as_ref()) {
        Ok(normalized)
            if mode == NativeLoweringMode::CountOnly
                && !normalized.aggregate_root().is_some_and(|root| {
                    normalized.single_source_input().is_some_and(|source| {
                        native_zero_column_count_source_is_safe(
                            source.filter,
                            root.pre_aggregate_projection,
                            root.aggregate,
                        )
                    })
                }) =>
        {
            let mut report = PlanningReport::new();
            report.push_rejected(normalized.candidate.clone(), crate::policy::FallbackReason::UnsupportedOperator);
            NativeLoweringResult {
                plans: Vec::new(),
                report,
                source_diagnostics: source_diagnostics_for_normalized_plan(&normalized),
            }
        },
        Ok(normalized) => lower_normalized_plan(&normalized).map_or_else(
            |reason| {
                let mut report = PlanningReport::new();
                report.push_rejected(normalized.candidate.clone(), reason);
                NativeLoweringResult {
                    plans: Vec::new(),
                    report,
                    source_diagnostics: source_diagnostics_for_normalized_plan(&normalized),
                }
            },
            |native_plan| {
                let source_diagnostics = source_diagnostics_for_native_plan(&native_plan);
                let mut report = PlanningReport::new();
                report.push_rewritten(normalized.candidate.clone());
                NativeLoweringResult { plans: vec![native_plan], report, source_diagnostics }
            },
        ),
        Err(rejection) => {
            let mut report = PlanningReport::new();
            report.push_rejected(rejection.candidate, rejection.reason);
            NativeLoweringResult { plans: Vec::new(), report, source_diagnostics: Vec::new() }
        },
    }
}

fn source_diagnostics_for_native_plan(plan: &NativeQueryPlan) -> Vec<NativeSourceDiagnostics> {
    plan.nodes()
        .iter()
        .filter_map(|node| match &node.kind {
            NativePlanNode::Source(source) => match source.as_ref() {
                NativeSourceNode::LocalParquet(source) => Some(source.diagnostics.clone()),
                NativeSourceNode::GpuResident(_) => None,
            },
            _ => None,
        })
        .collect()
}
