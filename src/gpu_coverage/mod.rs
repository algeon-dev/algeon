//! Query-level GPU coverage validation.
//!
//! This module answers the developer-facing planning question: whether the
//! optimized physical path contains known DataFusion CPU execution. It is a
//! plan-time validation surface, not a runtime admission or success guarantee.

mod explain_gpu;
mod model;
mod physical;
mod preflight;
mod recursion_guard;
mod remedy;

#[cfg(test)]
#[path = "tests.rs"]
mod tests;

use std::sync::Arc;

use datafusion::{execution::SessionStateBuilder, prelude::SessionContext};

pub(crate) use self::explain_gpu::ExplainGpuOutputMode;
#[cfg(feature = "server")]
pub(crate) use self::explain_gpu::parse_explain_gpu;
pub use self::model::{
    QueryGpuCandidateOutcome, QueryGpuCandidateReasonKind, QueryGpuCandidateSelectionOutcome, QueryGpuCoverage,
    QueryGpuCoverageConfidence, QueryGpuCoverageDomains, QueryGpuCoverageOverall, QueryGpuCugraphCall,
    QueryGpuRejectionPhase, QueryGpuRemedy, QueryGpuRemedyKind, QueryGpuRuntimeCaveat, QueryGpuValidatedUnder,
};
use self::{
    physical::{candidate_outcomes_from_report, cugraph_calls, planning_diagnostics_did_io},
    preflight::{
        QueryGpuRejection, classify_physical_planning_error, mark_coverage_validation_in_progress,
        optimizer_config_for_validation, planning_did_io_for_physical_error, preflight_single_query, validated_under,
    },
};
use crate::{
    final_plan_observation::observe_final_plan, planner::NexusNativePhysicalOptimizerRule,
    session::NexusSessionStateBuilderExt,
};

/// SQL table function that exposes query GPU coverage validation.
pub const NEXUS_EXPLAIN_COVERAGE_FUNCTION: &str = "nexus_explain_coverage";
/// Version of the shared multi-row UDTF and `EXPLAIN GPU` output schema.
pub const EXPLAIN_GPU_SCHEMA_VERSION: u8 = 1;
pub(crate) const COVERAGE_VALIDATOR_RECURSION_REASON_CODE: &str = "coverage_validator_recursion_unsupported";
pub(crate) const GPU_COVERAGE_REJECTION_REASON_FACT: &str = "gpu_coverage_rejection_reason";

pub(crate) use self::preflight::{coverage_validator_recursion_error, gpu_coverage_validation_in_progress};

/// Validate a single SQL query against the native GPU planning path.
///
/// The validation clones the caller's planning state. When the caller has an
/// installed native optimizer rule, validation uses an isolated retained copy
/// of that exact configuration, including completed-plan requirements. A
/// caller with no native rule remains on its DataFusion CPU baseline.
///
/// Candidate outcomes are retained as optimizer diagnostics, not an alternative
/// verdict. They can be non-empty even when the final optimized plan has
/// [`QueryGpuCoverageOverall::NoCpuInPlannedPath`].
///
/// `validated_under.catalog_snapshot` is a constant label for the cloned
/// planning state, not a catalog or table-state fingerprint. Do not compare
/// coverage verdicts across catalog, view, or table changes solely by this
/// field.
pub async fn validate_query(ctx: &SessionContext, sql: impl AsRef<str>) -> crate::error::Result<QueryGpuCoverage> {
    let sql = sql.as_ref();
    let mut planning_state_builder = SessionStateBuilder::new_from_existing(ctx.state());
    mark_coverage_validation_in_progress(&mut planning_state_builder);
    let planning_state = planning_state_builder.build();
    let config = optimizer_config_for_validation(&planning_state)?;
    let native_rule = config
        .optimizer_config
        .as_ref()
        .map(|optimizer_config| Arc::new(NexusNativePhysicalOptimizerRule::new(optimizer_config.clone())));
    let planning_state = if let Some(native_rule) = &native_rule {
        SessionStateBuilder::new_from_existing(planning_state).with_nexus_native_rule(Arc::clone(native_rule)).build()
    } else {
        planning_state
    };
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
            let diagnostics = native_rule.as_ref().and_then(|rule| rule.last_planning_diagnostics());
            let phase = classify_physical_planning_error(&error);
            let planning_did_io = planning_did_io_for_physical_error(phase, diagnostics.as_ref());
            return Ok(QueryGpuCoverage::rejected_with_diagnostics(
                validated_under,
                QueryGpuRejection::from_datafusion_error(phase, &error),
                diagnostics.as_ref(),
                planning_did_io,
            ));
        },
    };

    let diagnostics = native_rule.as_ref().and_then(|rule| rule.last_planning_diagnostics());
    let observation = observe_final_plan(plan.as_ref());
    let candidate_outcomes =
        diagnostics.as_ref().map(|diagnostics| candidate_outcomes_from_report(&diagnostics.report)).unwrap_or_default();
    let cugraph_calls = cugraph_calls(plan.as_ref());
    let planning_did_io = diagnostics.as_ref().is_some_and(planning_diagnostics_did_io);
    Ok(QueryGpuCoverage::planned(
        validated_under,
        plan.as_ref(),
        observation,
        candidate_outcomes,
        cugraph_calls,
        planning_did_io,
        diagnostics.as_ref(),
    ))
}
