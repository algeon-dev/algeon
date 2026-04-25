//! Query-level GPU coverage validation.
//!
//! This module answers the developer-facing planning question: whether the
//! optimized physical path contains known DataFusion CPU execution. It is a
//! plan-time validation surface, not a runtime admission or success guarantee.

mod model;
mod physical;
mod preflight;
mod recursion_guard;

#[cfg(test)]
#[path = "tests.rs"]
mod tests;

use std::sync::Arc;

use datafusion::{execution::SessionStateBuilder, prelude::SessionContext};

pub use self::model::{
    QueryGpuCoverage, QueryGpuCoverageConfidence, QueryGpuCoverageDomains, QueryGpuCoverageOverall,
    QueryGpuCudfNativeCoverage, QueryGpuCugraphCall, QueryGpuCugraphOperatorDomain, QueryGpuFallbackReason,
    QueryGpuFallbackReasonKind, QueryGpuPresence, QueryGpuRejectionPhase, QueryGpuRuntimeCaveat,
    QueryGpuValidatedUnder,
};
use self::{
    physical::{fallback_reasons_from_report, planning_diagnostics_did_io, walk_gpu_coverage},
    preflight::{
        QueryGpuRejection, classify_physical_planning_error, mark_coverage_validation_in_progress,
        optimizer_config_for_validation, planning_did_io_for_physical_error, preflight_single_query, validated_under,
    },
};
use crate::{
    native_report::post_native_plan_coverage, planner::CudfNativePhysicalOptimizerRule,
    session::CudfSessionStateBuilderExt,
};

/// SQL table function that exposes query GPU coverage validation.
pub const NEXUS_EXPLAIN_COVERAGE_FUNCTION: &str = "nexus_explain_coverage";
pub(crate) const COVERAGE_VALIDATOR_RECURSION_REASON_CODE: &str = "coverage_validator_recursion_unsupported";
pub(crate) const GPU_COVERAGE_REJECTION_REASON_FACT: &str = "gpu_coverage_rejection_reason";

pub(crate) use self::preflight::{coverage_validator_recursion_error, gpu_coverage_validation_in_progress};

/// Validate a single SQL query against the native GPU planning path.
///
/// The validation uses an isolated retained native optimizer rule installed
/// into a derived [`SessionContext`]. It plans as-if [`crate::policy::GpuFallbackPolicy::GpuPreferred`]
/// to preserve fallback diagnostics, while recording the caller's actual policy
/// in [`QueryGpuCoverage::validated_under`].
///
/// `fallback_reasons` are retained per-candidate optimizer diagnostics, not an
/// alternative verdict. They can be non-empty even when the final optimized plan
/// has [`QueryGpuCoverageOverall::NoCpuInPlannedPath`].
///
/// `validated_under.catalog_snapshot` is a constant label for the cloned
/// planning state, not a catalog or table-state fingerprint. Do not compare
/// coverage verdicts across catalog, view, or table changes solely by this
/// field.
pub async fn validate_query(ctx: &SessionContext, sql: impl AsRef<str>) -> crate::error::Result<QueryGpuCoverage> {
    let sql = sql.as_ref();
    let config = optimizer_config_for_validation(ctx);
    let native_rule = Arc::new(CudfNativePhysicalOptimizerRule::new(config.diagnostic_optimizer_config.clone()));
    let mut planning_state_builder =
        SessionStateBuilder::new_from_existing(ctx.state()).with_cudf_native_rule(Arc::clone(&native_rule));
    mark_coverage_validation_in_progress(&mut planning_state_builder);
    let planning_state = planning_state_builder.build();
    let validated_under = validated_under(&planning_state, &config);
    if let Some(rejection) = preflight_single_query(&planning_state, sql) {
        return Ok(QueryGpuCoverage::rejected(validated_under, rejection));
    }
    let planning_ctx = SessionContext::new_with_state(planning_state);

    let df = match planning_ctx.sql(sql).await {
        Ok(df) => df,
        Err(error) => {
            return Ok(QueryGpuCoverage::rejected(
                validated_under,
                QueryGpuRejection::from_datafusion_error(QueryGpuRejectionPhase::LogicalPlanning, &error),
            ));
        },
    };
    let plan = match df.create_physical_plan().await {
        Ok(plan) => plan,
        Err(error) => {
            let diagnostics = native_rule.last_planning_diagnostics();
            let phase = classify_physical_planning_error(&error);
            let planning_did_io = planning_did_io_for_physical_error(phase, diagnostics.as_ref());
            return Ok(QueryGpuCoverage::rejected_with_diagnostics(
                validated_under,
                QueryGpuRejection::from_datafusion_error(phase, &error),
                diagnostics.as_ref().map(|diagnostics| &diagnostics.report),
                planning_did_io,
            ));
        },
    };

    let diagnostics = native_rule.last_planning_diagnostics();
    let cudf_coverage = post_native_plan_coverage(plan.as_ref());
    let walk = walk_gpu_coverage(plan.as_ref());
    let fallback_reasons =
        diagnostics.as_ref().map(|diagnostics| fallback_reasons_from_report(&diagnostics.report)).unwrap_or_default();
    let planning_did_io = diagnostics.as_ref().is_some_and(planning_diagnostics_did_io);
    Ok(QueryGpuCoverage::planned(
        validated_under,
        plan.as_ref(),
        cudf_coverage.execution_path,
        walk,
        fallback_reasons,
        planning_did_io,
    ))
}
