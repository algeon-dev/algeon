mod literal;
mod lowering;
mod normalize;
mod pruning;
mod source;
mod validate;

use std::sync::Arc;

use cudf_query_engine::{QueryPlan as NativeQueryPlan, SourceDiagnostics as NativeSourceDiagnostics};
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
                && !native_zero_column_count_source_is_safe(
                    normalized.filter,
                    normalized.pre_aggregate_projection,
                    normalized.aggregate,
                ) =>
        {
            let mut report = PlanningReport::new();
            report.push_rejected(normalized.candidate, crate::policy::FallbackReason::UnsupportedOperator);
            NativeLoweringResult {
                plans: Vec::new(),
                report,
                source_diagnostics: source_diagnostics_for_normalized_plan(&normalized),
            }
        },
        Ok(normalized) => lower_normalized_plan(&normalized).map_or_else(
            |reason| {
                let mut report = PlanningReport::new();
                report.push_rejected(normalized.candidate, reason);
                NativeLoweringResult {
                    plans: Vec::new(),
                    report,
                    source_diagnostics: source_diagnostics_for_normalized_plan(&normalized),
                }
            },
            |native_plan| {
                let mut report = PlanningReport::new();
                report.push_rewritten(normalized.candidate);
                NativeLoweringResult {
                    plans: vec![native_plan],
                    report,
                    source_diagnostics: source_diagnostics_for_normalized_plan(&normalized),
                }
            },
        ),
        Err(rejection) => {
            let mut report = PlanningReport::new();
            report.push_rejected(rejection.candidate, rejection.reason);
            NativeLoweringResult { plans: Vec::new(), report, source_diagnostics: Vec::new() }
        },
    }
}
