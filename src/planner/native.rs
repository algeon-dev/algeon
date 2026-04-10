use std::sync::{Arc, Mutex};

use datafusion::{
    common::{Result as DataFusionResult, config::ConfigOptions},
    physical_optimizer::PhysicalOptimizerRule,
    physical_plan::ExecutionPlan,
};
use nexus_query_engine::QueryId;

use super::CudfOptimizerConfig;
use crate::{
    Error, ErrorCode,
    native_exec::{NativeCudfComposedExec, NativeCudfExec},
    native_report::{NativeJoinDiagnosticSummary, physical_join_diagnostics},
    physical_plan::physical_plan_shape,
    report::PlanningReport,
    session::{CUDF_NATIVE_COUNT_ONLY_PHYSICAL_OPTIMIZER_RULE_NAME, CUDF_NATIVE_PHYSICAL_OPTIMIZER_RULE_NAME},
};

mod adaptive;
mod attempt;
mod strict_bounded;

pub use adaptive::{NativeAdaptivePipelineCandidateDiagnostics, NativeAdaptivePipelineDiagnostics};
use attempt::{
    apply_native_fallback_policy, optimize_count_only_native_plan, optimize_final_native_plan,
    peel_cooperative_native_exec,
};
pub use strict_bounded::{
    NativeStrictBoundedAcceptedCandidateDiagnostics, NativeStrictBoundedCompiledGraphDiagnostics,
    NativeStrictBoundedContractDiagnostics, NativeStrictBoundedRejectedCandidateDiagnostics,
    STRICT_BOUNDED_NON_JOIN_OUTSIDE_REPORT_SCOPE_REASON,
};

/// Last-observed native planning diagnostics from a retained optimizer rule.
///
/// The strict bounded and adaptive fields are intentionally separate. Report
/// code should not project adaptive evidence into strict contract rows, or
/// strict preflight failures into adaptive runtime history.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativePlanningDiagnostics {
    /// Root DataFusion execution node name seen before native optimization.
    pub input_plan_root: String,
    /// Stable textual shape of the input physical plan used for report context.
    pub input_plan_shape: String,
    /// Join diagnostics captured before native replacement can remove
    /// DataFusion join nodes from the post-native plan.
    pub input_join_diagnostics: Vec<NativeJoinDiagnosticSummary>,
    /// Candidate rewrite outcomes from the native physical optimizer rule.
    pub report: PlanningReport,
    /// DataFusion-free native optimizer diagnostics from the lowered plan.
    pub native_optimizer: nexus_query_engine::NativeOptimizerDiagnostics,
    /// Source-read diagnostics captured while lowering/planning native sources.
    pub source_diagnostics: Vec<nexus_query_engine::SourceDiagnostics>,
    /// Strict bounded contract evidence; never populated from adaptive rows.
    pub strict_bounded_contract: NativeStrictBoundedContractDiagnostics,
    /// Adaptive admission evidence; never projected into strict contract rows.
    pub adaptive_pipeline: NativeAdaptivePipelineDiagnostics,
}

/// DataFusion physical optimizer rule that rewrites supported plans to native cuDF execution.
///
/// The rule is reusable across sessions when callers want to inspect
/// [`last_planning_diagnostics`](Self::last_planning_diagnostics). Session
/// insertion deduplicates by the public rule names in [`crate::session`].
/// `GpuPreferred` leaves rejected candidates on the original DataFusion plan;
/// `GpuOnly` reports the first native rejection as an optimizer error.
#[derive(Debug)]
pub struct CudfNativePhysicalOptimizerRule {
    config: CudfOptimizerConfig,
    stage: NativeOptimizerStage,
    state: Arc<CudfNativePhysicalOptimizerRuleState>,
}

#[derive(Debug)]
struct CudfNativePhysicalOptimizerRuleState {
    last_input_plan_root: Mutex<Option<String>>,
    last_input_plan_shape: Mutex<Option<String>>,
    last_input_join_diagnostics: Mutex<Vec<NativeJoinDiagnosticSummary>>,
    last_report: Mutex<Option<PlanningReport>>,
    last_native_optimizer: Mutex<nexus_query_engine::NativeOptimizerDiagnostics>,
    last_source_diagnostics: Mutex<Vec<nexus_query_engine::SourceDiagnostics>>,
    last_strict_bounded_contract: Mutex<NativeStrictBoundedContractDiagnostics>,
    last_adaptive_pipeline: Mutex<NativeAdaptivePipelineDiagnostics>,
}

// Distinct rule identities let the count-only probe run before the final rewrite pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NativeOptimizerStage {
    // Dedicated pass for plans where output-requirement peeling exposes a native count.
    CountOnlyProbe,
    // Full native lowering, fallback handling, and fragment composition.
    Final,
}

impl CudfNativePhysicalOptimizerRule {
    /// Create the final native rewrite rule.
    pub fn new(config: CudfOptimizerConfig) -> Self {
        Self {
            config,
            stage: NativeOptimizerStage::Final,
            state: Arc::new(CudfNativePhysicalOptimizerRuleState {
                last_input_plan_root: Mutex::new(None),
                last_input_plan_shape: Mutex::new(None),
                last_input_join_diagnostics: Mutex::new(Vec::new()),
                last_report: Mutex::new(None),
                last_native_optimizer: Mutex::new(nexus_query_engine::NativeOptimizerDiagnostics::default()),
                last_source_diagnostics: Mutex::new(Vec::new()),
                last_strict_bounded_contract: Mutex::new(NativeStrictBoundedContractDiagnostics::default()),
                last_adaptive_pipeline: Mutex::new(NativeAdaptivePipelineDiagnostics::default()),
            }),
        }
    }

    /// Return the immutable config copied into rewritten native exec nodes.
    pub fn config(&self) -> &CudfOptimizerConfig {
        &self.config
    }

    pub(crate) fn count_only_probe_rule(&self) -> Self {
        Self {
            config: self.config.clone(),
            stage: NativeOptimizerStage::CountOnlyProbe,
            state: Arc::clone(&self.state),
        }
    }

    /// Most recent planning report produced by this retained rule, if it has run.
    pub fn last_report(&self) -> Option<PlanningReport> {
        match self.state.last_report.lock() {
            Ok(report) => report.clone(),
            Err(poisoned) => {
                #[cfg(feature = "server")]
                tracing::warn!("native planning report mutex was poisoned; returning the last captured report");
                poisoned.into_inner().clone()
            },
        }
    }

    /// Source diagnostics captured during the most recent native planning attempt.
    pub fn last_source_diagnostics(&self) -> Vec<nexus_query_engine::SourceDiagnostics> {
        self.state.last_source_diagnostics.lock().expect("native source diagnostics mutex poisoned").clone()
    }

    /// Structured diagnostics from the most recent run of this retained rule.
    ///
    /// The result is overwritten on each optimizer invocation and is intended
    /// for report harnesses and tests, not as a historical log.
    pub fn last_planning_diagnostics(&self) -> Option<NativePlanningDiagnostics> {
        self.last_report().map(|report| NativePlanningDiagnostics {
            input_plan_root: self
                .state
                .last_input_plan_root
                .lock()
                .expect("native input plan root mutex poisoned")
                .clone()
                .unwrap_or_default(),
            input_plan_shape: self
                .state
                .last_input_plan_shape
                .lock()
                .expect("native input plan shape mutex poisoned")
                .clone()
                .unwrap_or_default(),
            input_join_diagnostics: self
                .state
                .last_input_join_diagnostics
                .lock()
                .expect("native input join diagnostics mutex poisoned")
                .clone(),
            report,
            native_optimizer: self
                .state
                .last_native_optimizer
                .lock()
                .expect("native optimizer diagnostics mutex poisoned")
                .clone(),
            source_diagnostics: self.last_source_diagnostics(),
            strict_bounded_contract: self
                .state
                .last_strict_bounded_contract
                .lock()
                .expect("native strict bounded contract diagnostics mutex poisoned")
                .clone(),
            adaptive_pipeline: self
                .state
                .last_adaptive_pipeline
                .lock()
                .expect("native adaptive pipeline diagnostics mutex poisoned")
                .clone(),
        })
    }
}

impl PhysicalOptimizerRule for CudfNativePhysicalOptimizerRule {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        config: &ConfigOptions,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        if plan.downcast_ref::<NativeCudfExec>().is_some() || plan.downcast_ref::<NativeCudfComposedExec>().is_some() {
            return Ok(plan);
        }
        if let Some(native_plan) = peel_cooperative_native_exec(&plan) {
            return Ok(native_plan);
        }

        *self.state.last_input_plan_root.lock().expect("native input plan root mutex poisoned") =
            Some(plan.name().to_string());
        *self.state.last_input_plan_shape.lock().expect("native input plan shape mutex poisoned") =
            Some(physical_plan_shape(plan.as_ref()));
        *self.state.last_input_join_diagnostics.lock().expect("native input join diagnostics mutex poisoned") =
            physical_join_diagnostics(plan.as_ref());

        let effective_config =
            self.config.with_datafusion_config_options(config)?.with_query_id_if_absent(QueryId::mint());
        let native_execution_config =
            effective_config.native_execution_config().clone().with_plan_query_admission_ticket();
        let effective_config = effective_config.with_native_execution_config(native_execution_config);
        let native_execution_config = effective_config.native_execution_config();
        let native_optimizer_options = effective_config.native_optimizer_options();
        let native_plan_preference_options = effective_config.native_plan_preference_options();

        let optimized = match self.stage {
            NativeOptimizerStage::CountOnlyProbe => optimize_count_only_native_plan(
                Arc::clone(&plan),
                native_execution_config,
                native_optimizer_options,
                native_plan_preference_options,
            )?,
            NativeOptimizerStage::Final => optimize_final_native_plan(
                Arc::clone(&plan),
                native_execution_config,
                native_optimizer_options,
                native_plan_preference_options,
            )?,
        };
        let optimized = apply_native_fallback_policy(Arc::clone(&plan), optimized, &effective_config)?;
        *self.state.last_report.lock().map_err(|_| {
            datafusion::common::DataFusionError::from(Error::new(
                ErrorCode::Internal,
                "datafusion_nexus.native_optimizer.report",
                "native planning report mutex was poisoned",
            ))
        })? = Some(optimized.report.clone());
        *self.state.last_native_optimizer.lock().expect("native optimizer diagnostics mutex poisoned") =
            optimized.native_optimizer.clone();
        *self.state.last_source_diagnostics.lock().expect("native source diagnostics mutex poisoned") =
            optimized.source_diagnostics.clone();
        *self
            .state
            .last_strict_bounded_contract
            .lock()
            .expect("native strict bounded contract diagnostics mutex poisoned") =
            optimized.strict_bounded_contract.clone();
        *self.state.last_adaptive_pipeline.lock().expect("native adaptive pipeline diagnostics mutex poisoned") =
            optimized.adaptive_pipeline.clone();
        Ok(optimized.plan)
    }

    fn name(&self) -> &str {
        match self.stage {
            NativeOptimizerStage::CountOnlyProbe => CUDF_NATIVE_COUNT_ONLY_PHYSICAL_OPTIMIZER_RULE_NAME,
            NativeOptimizerStage::Final => CUDF_NATIVE_PHYSICAL_OPTIMIZER_RULE_NAME,
        }
    }

    fn schema_check(&self) -> bool {
        true
    }
}
