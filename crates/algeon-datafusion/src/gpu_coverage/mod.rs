//! Query-level GPU coverage validation.
//!
//! This module answers the developer-facing planning question: whether the
//! optimized physical path contains known DataFusion CPU execution. It is a
//! plan-time diagnosis surface only, not runtime admission, not an execution
//! verdict, and not sealed-attempt retry advice.
//!
//! Each validation produces one [`QueryGpuPlanningOutcome`]: either a planned
//! [`FinalPlanCoverage`] describing the single completed executable plan, or
//! a typed rejection recording where planning failed. Candidate diagnostics
//! stay per-candidate under [`crate::report::CandidateOutcome`]; attempt
//! termination stays runtime-owned under [`crate::attempt`].

mod explain_gpu;
mod model;
mod physical;
mod preflight;
mod recursion_guard;
mod remedy;
mod sql;

#[cfg(feature = "server")]
pub(crate) use sql::{coverage_batch, coverage_schema};

#[cfg(test)]
mod tests;

use datafusion::{execution::SessionStateBuilder, prelude::SessionContext};
use futures::future::BoxFuture;

pub use self::{
    explain_gpu::{ExplainGpuOutputMode, ExplainGpuStatement, parse_explain_gpu},
    model::{
        QueryGpuCandidateOutcome, QueryGpuCandidateReasonKind, QueryGpuCoverage, QueryGpuCugraphCall,
        QueryGpuPlanningOutcome, QueryGpuPlanningRejection, QueryGpuRejectionPhase, QueryGpuRemedy, QueryGpuRemedyKind,
        QueryGpuRuntimeCaveat, QueryGpuValidatedUnder,
    },
};
use self::{
    physical::{candidate_outcomes_from_report, cugraph_calls},
    preflight::{
        QueryGpuRejection, classify_physical_planning_error, mark_coverage_validation_in_progress,
        optimizer_config_for_validation, planning_did_io_for_physical_error, preflight_single_query, validated_under,
    },
};
use crate::{error::Result, final_plan_observation::observe_final_plan, planner::native::Invocation};
pub use crate::{planner::FinalPlanCoverage, report::CandidateOutcomeKind};

/// Metadata UDTF that exposes query GPU coverage validation.
pub const ALGEON_EXPLAIN_COVERAGE_FUNCTION: &str = "algeon_explain_coverage";
pub(crate) const COVERAGE_VALIDATOR_RECURSION_REASON_CODE: &str = "coverage_validator_recursion_unsupported";
pub(crate) const GPU_COVERAGE_REJECTION_REASON_FACT: &str = "gpu_coverage_rejection_reason";

pub(crate) use self::{
    preflight::{coverage_validator_recursion_error, gpu_coverage_validation_in_progress},
    sql::gpu_coverage_table_function,
};

/// Validate a single SQL query against the native GPU planning path.
///
/// The validation clones the caller's planning state and plans under its own
/// invocation, detached from any attempt the caller's state belongs to, so it
/// never binds that attempt's admission or disposition. The caller must have
/// an installed native optimizer rule; validation plans with that rule and its
/// completed-plan requirements.
///
/// Candidate outcomes are retained as optimizer diagnostics, not an alternative
/// verdict. One report can hold several decision kinds at once; they never
/// re-derive the completed plan's disposition.
///
/// `validated_under.catalog_snapshot` is a constant label for the cloned
/// planning state, not a catalog or table-state fingerprint. Do not compare
/// coverage verdicts across catalog, view, or table changes solely by this
/// field.
pub fn validate_query<'a>(
    ctx: &'a SessionContext,
    sql: impl AsRef<str> + Send + 'a,
) -> BoxFuture<'a, Result<QueryGpuCoverage>> {
    Box::pin(async move {
        let sql = sql.as_ref();
        let invocation = Invocation::detached();
        let mut planning_state_builder = SessionStateBuilder::new_from_existing(ctx.state());
        mark_coverage_validation_in_progress(&mut planning_state_builder);
        invocation.install(planning_state_builder.config().get_or_insert_with(Default::default));
        let planning_state = planning_state_builder.build();
        let config = optimizer_config_for_validation(&planning_state)?;
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
                let diagnostics = invocation.planning_diagnostics();
                let phase = classify_physical_planning_error(&error);
                let planning_did_io = planning_did_io_for_physical_error(phase);
                return Ok(QueryGpuCoverage::rejected_with_diagnostics(
                    validated_under,
                    QueryGpuRejection::from_datafusion_error(phase, &error),
                    diagnostics.as_ref(),
                    planning_did_io,
                ));
            },
        };

        let diagnostics = invocation.planning_diagnostics();
        let observation = observe_final_plan(plan.as_ref());
        let candidate_outcomes = diagnostics
            .as_ref()
            .map(|diagnostics| candidate_outcomes_from_report(&diagnostics.report))
            .unwrap_or_default();
        let cugraph_calls = cugraph_calls(plan.as_ref());
        Ok(QueryGpuCoverage::planned(
            validated_under,
            plan.as_ref(),
            observation,
            candidate_outcomes,
            cugraph_calls,
            diagnostics.as_ref(),
        ))
    })
}
