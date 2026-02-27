//! Final-stage guard for adapter-owned GPU-only table-format scan leaves.
//!
//! [`CudfTableFormatScanExec`] is a metadata-only planning leaf. Native
//! lowering is expected to consume it; if the native physical optimizer rule
//! leaves it in place, the surrounding plan cannot fall back to DataFusion CPU
//! execution because the leaf has no CPU Arrow reader. Letting such a plan
//! reach `ExecutionPlan::execute` produces a placeholder runtime error.
//!
//! This rule runs immediately after the native rewrite rule. If any
//! `CudfTableFormatScanExec` survives, the rule turns that into a structured
//! planning error tagged with [`FallbackReason::TableFormatGpuOnlyLeafNotRewritten`]
//! so callers see a real native rejection reason rather than the placeholder.

use std::sync::Arc;

use datafusion::{
    common::{Result as DataFusionResult, config::ConfigOptions},
    physical_optimizer::PhysicalOptimizerRule,
    physical_plan::ExecutionPlan,
};

use crate::{
    error::{Error, ErrorCode},
    planner::CudfNativePhysicalOptimizerRule,
    policy::FallbackReason,
    report::{CandidateOutcome, PlanningReport},
    table_format::CudfTableFormatScanExec,
};

/// Public rule name and deduplication key for the leftover GPU-only leaf guard.
pub const CUDF_TABLE_FORMAT_GPU_ONLY_LEAF_GUARD_RULE_NAME: &str = "datafusion_nexus_table_format_gpu_only_leaf_guard";

/// Final-stage physical optimizer rule that converts leftover
/// [`CudfTableFormatScanExec`] nodes into a structured planning error.
///
/// The rule reads diagnostics from a sibling [`CudfNativePhysicalOptimizerRule`]
/// when wired through session installation so the resulting error can quote
/// the first native rejection. Constructed standalone with [`Self::new`], the
/// rule still fires but the `first_native_rejection` fact stays empty.
#[derive(Debug, Clone)]
pub struct CudfTableFormatGpuOnlyLeafGuardRule {
    native_rule: Option<Arc<CudfNativePhysicalOptimizerRule>>,
}

impl CudfTableFormatGpuOnlyLeafGuardRule {
    /// Build a guard rule that does not reference a native rule's diagnostics.
    pub fn new() -> Self {
        Self { native_rule: None }
    }

    /// Build a guard rule that quotes the first native rejection seen by the
    /// retained native rule into emitted errors.
    pub fn with_native_rule(native_rule: Arc<CudfNativePhysicalOptimizerRule>) -> Self {
        Self { native_rule: Some(native_rule) }
    }
}

impl Default for CudfTableFormatGpuOnlyLeafGuardRule {
    fn default() -> Self {
        Self::new()
    }
}

impl PhysicalOptimizerRule for CudfTableFormatGpuOnlyLeafGuardRule {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let Some(leaf) = find_leftover_table_format_scan(plan.as_ref()) else {
            return Ok(plan);
        };
        let first_native_rejection = self
            .native_rule
            .as_ref()
            .and_then(|rule| rule.last_report())
            .as_ref()
            .and_then(first_rejection_code)
            .map(|code| code.to_string());
        Err(leftover_leaf_error(leaf, plan.as_ref(), first_native_rejection).into())
    }

    fn name(&self) -> &str {
        CUDF_TABLE_FORMAT_GPU_ONLY_LEAF_GUARD_RULE_NAME
    }

    fn schema_check(&self) -> bool {
        false
    }
}

fn find_leftover_table_format_scan(plan: &dyn ExecutionPlan) -> Option<&CudfTableFormatScanExec> {
    if let Some(scan) = plan.as_any().downcast_ref::<CudfTableFormatScanExec>() {
        return Some(scan);
    }
    for child in plan.children() {
        if let Some(scan) = find_leftover_table_format_scan(child.as_ref()) {
            return Some(scan);
        }
    }
    None
}

fn first_rejection_code(report: &PlanningReport) -> Option<&'static str> {
    report.outcomes().iter().find_map(|outcome| match outcome {
        CandidateOutcome::Rejected { reason, .. } => Some(reason.code()),
        CandidateOutcome::Rewritten { .. } => None,
    })
}

fn leftover_leaf_error(
    leaf: &CudfTableFormatScanExec,
    plan: &dyn ExecutionPlan,
    first_native_rejection: Option<String>,
) -> Error {
    let diagnostics = leaf.diagnostics();
    let mut error = Error::new(
        ErrorCode::NativeCandidateRejected,
        "datafusion_nexus.planner.table_format_gpu_only_leaf_guard",
        "table-format scan leaf was not rewritten to native and has no CPU Arrow reader fallback",
    )
    .with_fact("fallback_reason", FallbackReason::TableFormatGpuOnlyLeafNotRewritten.code())
    .with_fact("leftover_operator", leaf.name())
    .with_fact("table_format", diagnostics.table_format)
    .with_fact("cpu_arrow_reader_fallback", false)
    .with_fact("physical_plan_root", plan.name())
    .with_fact("physical_plan_shape", physical_plan_shape(plan));
    if let Some(reason) = first_native_rejection {
        error = error.with_fact("first_native_rejection", reason);
    }
    error
}

fn physical_plan_shape(plan: &dyn ExecutionPlan) -> String {
    let children = plan.children();
    if children.is_empty() {
        return plan.name().to_string();
    }

    let mut shape = String::new();
    shape.push_str(plan.name());
    shape.push('(');
    for (index, child) in children.iter().enumerate() {
        if index > 0 {
            shape.push(',');
        }
        shape.push_str(&physical_plan_shape(child.as_ref()));
    }
    shape.push(')');
    shape
}
