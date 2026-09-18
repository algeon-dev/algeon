use std::sync::{Arc, Mutex};

use datafusion::{
    common::{DataFusionError, Result as DataFusionResult, config::ConfigOptions},
    physical_optimizer::PhysicalOptimizerRule,
    physical_plan::{ChildrenPropertiesMode, ExecutionPlan, ReplaceChildrenOptions},
};
use nexus_query_engine::{
    admission::{AttemptCapabilityEvidence, AttemptExecutionManifest, AttemptExecutionOccurrence},
    metrics::QueryId,
};

use super::{DataFusionCpuExecutionPolicy, FinalPlanDisposition, NexusNativeOptimizerConfig, classify_final_plan};
use crate::{
    attempt::NexusAttemptExtension,
    error::{Error, ErrorCode},
    native::lakehouse_pruning,
    native_exec::{
        GpuFunctionHostOccurrenceExec, NexusNativeComposedExec, NexusNativeComposedGpuProducerExec, NexusNativeExec,
        NexusNativeExecutionConfig, NexusNativeGpuProducerExec,
        admission::QueryAdmissionTicket,
        graph::{GraphExecutionPlan, graph_execution_plan},
        plan_is_query_admission_wrapped, wrap_plan_with_query_admission_ticket,
    },
    native_report::{NativeJoinDiagnosticSummary, physical_join_diagnostics},
    physical_plan::physical_plan_shape,
    report::{CandidateOutcome, PlanningReport},
    session::{NEXUS_NATIVE_COUNT_ONLY_PHYSICAL_OPTIMIZER_RULE_NAME, NEXUS_NATIVE_PHYSICAL_OPTIMIZER_RULE_NAME},
};

mod attempt;
mod capability_envelope;
mod mixed;

use attempt::{
    NativeOptimizationAttempt, optimize_count_only_native_plan, optimize_final_native_plan,
    peel_cooperative_native_exec,
};
pub use capability_envelope::{
    CAPABILITY_NON_JOIN_OUTSIDE_REPORT_SCOPE_REASON, NativeCapabilityAcceptedCandidateDiagnostics,
    NativeCapabilityCompiledGraphDiagnostics, NativeCapabilityContractDiagnostics,
    NativeCapabilityDeferredCandidateDiagnostics, NativeCapabilityRejectedCandidateDiagnostics,
};
use mixed::is_certified_mixed_execution_plan;

/// Internal optimizer-rule identity for the completed-plan DataFusion CPU policy.
pub const NEXUS_FINAL_PLAN_REQUIREMENT_RULE_NAME: &str = "datafusion_nexus_final_plan_requirement";

/// Enforces the configured DataFusion CPU policy against the exact physical
/// plan produced by the preceding native rewrite and leaf-guard rules.
#[derive(Debug, Clone)]
pub struct NexusFinalPlanRequirementRule {
    config: NexusNativeOptimizerConfig,
    native_rule: Option<Arc<NexusNativePhysicalOptimizerRule>>,
}

impl NexusFinalPlanRequirementRule {
    /// Create a requirement gate without native-planning diagnostics.
    pub fn new(config: NexusNativeOptimizerConfig) -> Self {
        Self { config, native_rule: None }
    }

    pub(crate) fn with_native_rule(
        config: NexusNativeOptimizerConfig,
        native_rule: Arc<NexusNativePhysicalOptimizerRule>,
    ) -> Self {
        Self { config, native_rule: Some(native_rule) }
    }

    fn planning_report(&self) -> Option<PlanningReport> {
        self.native_rule.as_ref().and_then(|rule| rule.last_report())
    }
}

impl PhysicalOptimizerRule for NexusFinalPlanRequirementRule {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        if self.config.datafusion_cpu_execution_policy() == DataFusionCpuExecutionPolicy::Allow {
            return Ok(plan);
        }
        let disposition = classify_final_plan(plan.as_ref());
        // A feature-disabled cuGraph plan is a zero-child terminal capability
        // rejection. It executes neither a CPU fallback nor a GPU operator,
        // so it satisfies the no-CPU-execution contract without being
        // misclassified as a native workload.
        let feature_disabled_terminal = crate::cugraph_sql::is_feature_disabled_execution_plan(plan.as_ref());
        if disposition != FinalPlanDisposition::Native && !feature_disabled_terminal {
            return Err(
                final_plan_requirement_error(plan.as_ref(), disposition, self.planning_report().as_ref()).into()
            );
        }
        Ok(plan)
    }

    fn name(&self) -> &str {
        NEXUS_FINAL_PLAN_REQUIREMENT_RULE_NAME
    }

    fn schema_check(&self) -> bool {
        true
    }
}

fn final_plan_requirement_error(
    plan: &dyn ExecutionPlan,
    disposition: FinalPlanDisposition,
    report: Option<&PlanningReport>,
) -> Error {
    let rejection = report.and_then(first_native_not_supported);
    let message = rejection.and_then(|(_, _, detail)| detail).map_or_else(
        || "exact final physical plan does not satisfy the configured GPU execution requirement".to_string(),
        |detail| {
            format!(
                "GPU execution does not support {} column `{}` with Arrow type `{}`",
                detail.role, detail.name, detail.data_type
            )
        },
    );
    let mut error = Error::new(
        ErrorCode::FinalPlanRequirementUnsatisfied,
        "datafusion_nexus.planner.final_plan_requirement",
        message,
    )
    .with_fact("requirement", "no_datafusion_cpu")
    .with_fact("observed_final_plan_disposition", disposition.as_str())
    .with_fact("terminal_plan_disposition", "rejected")
    .with_fact("physical_plan_root", plan.name());
    if let Some((candidate, capability_reason, detail)) = rejection {
        error = error
            .with_fact("candidate_shape", candidate.shape.as_str())
            .with_fact("capability_reason", capability_reason.code())
            .with_fact("operator_count", candidate.operator_count);
        if let Some(detail) = detail {
            // `reason` is bounded operational classification and may be
            // logged. The column name, Arrow type, and role name the query
            // caller's own schema detail, so they stay ordinary facts hidden
            // from operational logs and Debug by default. The server
            // projection owner retains them explicitly for query-caller
            // surfaces (Flight metadata allowlist, query-stats capture) and
            // keeps them in the actionable status message.
            error = error
                .with_log_safe_fact("reason", capability_reason.code())
                .with_fact("unsupported_column", detail.name.as_str())
                .with_fact("unsupported_column_type", detail.data_type.as_str())
                .with_fact("unsupported_column_role", detail.role);
        }
    }
    error
}

fn first_native_not_supported(
    report: &PlanningReport,
) -> Option<(
    &crate::report::GpuCandidate,
    crate::capability::CapabilityReason,
    Option<&crate::report::UnsupportedColumnDetail>,
)> {
    report.outcomes().iter().enumerate().find_map(|(index, outcome)| match outcome {
        CandidateOutcome::NotSupported { candidate, capability_reason } => {
            Some((candidate, *capability_reason, report.unsupported_column(index)))
        },
        CandidateOutcome::Selected { .. }
        | CandidateOutcome::Deferred { .. }
        | CandidateOutcome::NotSelectedByCost { .. } => None,
    })
}

/// Last-observed native planning diagnostics from a retained optimizer rule.
///
/// Capability-contract evidence is retained from the planner. The optimizer
/// retains no alternate planning-history projection.
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
    pub native_optimizer: nexus_query_engine::optimizer::NativeOptimizerDiagnostics,
    /// Source-read diagnostics captured while lowering/planning native sources.
    pub source_diagnostics: Vec<nexus_query_engine::source::SourceDiagnostics>,
    /// Canonical capability-contract evidence.
    pub capability_contract: NativeCapabilityContractDiagnostics,
}

/// DataFusion physical optimizer rule that rewrites supported plans to native cuDF execution.
///
/// The rule is reusable across sessions when callers want to inspect
/// [`last_planning_diagnostics`](Self::last_planning_diagnostics). Session
/// insertion deduplicates by the public rule names in [`crate::session`].
/// Candidate misses leave the original DataFusion plan intact. A reject
/// DataFusion CPU policy runs in a later rule after native planning has
/// completed.
#[derive(Debug)]
pub struct NexusNativePhysicalOptimizerRule {
    config: NexusNativeOptimizerConfig,
    stage: NativeOptimizerStage,
    state: Arc<NexusNativePhysicalOptimizerRuleState>,
    mints_plan_ticket: bool,
}

#[derive(Debug)]
struct NexusNativePhysicalOptimizerRuleState {
    last_input_plan_root: Mutex<String>,
    last_input_plan_shape: Mutex<String>,
    last_input_join_diagnostics: Mutex<Vec<NativeJoinDiagnosticSummary>>,
    last_report: Mutex<Option<PlanningReport>>,
    last_native_optimizer: Mutex<nexus_query_engine::optimizer::NativeOptimizerDiagnostics>,
    last_source_diagnostics: Mutex<Vec<nexus_query_engine::source::SourceDiagnostics>>,
    last_capability_contract: Mutex<NativeCapabilityContractDiagnostics>,
}

// Distinct rule identities let the count-only probe run before the final rewrite pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NativeOptimizerStage {
    // Dedicated pass for plans where output-requirement peeling exposes a native count.
    CountOnlyProbe,
    // Full native lowering, fallback handling, and fragment composition.
    Final,
}

impl NexusNativePhysicalOptimizerRule {
    /// Create the final native rewrite rule.
    pub fn new(config: NexusNativeOptimizerConfig) -> Self {
        Self {
            config,
            stage: NativeOptimizerStage::Final,
            state: Arc::new(NexusNativePhysicalOptimizerRuleState {
                last_input_plan_root: Mutex::new(String::new()),
                last_input_plan_shape: Mutex::new(String::new()),
                last_input_join_diagnostics: Mutex::new(Vec::new()),
                last_report: Mutex::new(None),
                last_native_optimizer: Mutex::new(nexus_query_engine::optimizer::NativeOptimizerDiagnostics::default()),
                last_source_diagnostics: Mutex::new(Vec::new()),
                last_capability_contract: Mutex::new(NativeCapabilityContractDiagnostics::default()),
            }),
            mints_plan_ticket: true,
        }
    }

    #[cfg(feature = "cugraph")]
    pub(crate) fn new_cugraph_subplan(config: NexusNativeOptimizerConfig) -> Self {
        Self {
            config,
            stage: NativeOptimizerStage::Final,
            state: Arc::new(NexusNativePhysicalOptimizerRuleState {
                last_input_plan_root: Mutex::new(String::new()),
                last_input_plan_shape: Mutex::new(String::new()),
                last_input_join_diagnostics: Mutex::new(Vec::new()),
                last_report: Mutex::new(None),
                last_native_optimizer: Mutex::new(nexus_query_engine::optimizer::NativeOptimizerDiagnostics::default()),
                last_source_diagnostics: Mutex::new(Vec::new()),
                last_capability_contract: Mutex::new(NativeCapabilityContractDiagnostics::default()),
            }),
            mints_plan_ticket: false,
        }
    }

    /// Return the immutable config copied into rewritten native exec nodes.
    pub fn config(&self) -> &NexusNativeOptimizerConfig {
        &self.config
    }

    pub(crate) fn count_only_probe_rule(&self) -> Self {
        Self {
            config: self.config.clone(),
            stage: NativeOptimizerStage::CountOnlyProbe,
            state: Arc::clone(&self.state),
            mints_plan_ticket: self.mints_plan_ticket,
        }
    }

    /// Most recent planning report produced by this retained rule, if it has run.
    pub fn last_report(&self) -> Option<PlanningReport> {
        self.state.last_report.lock().expect("native planning report mutex poisoned").clone()
    }

    /// Source diagnostics captured during the most recent native planning attempt.
    pub fn last_source_diagnostics(&self) -> Vec<nexus_query_engine::source::SourceDiagnostics> {
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
                .clone(),
            input_plan_shape: self
                .state
                .last_input_plan_shape
                .lock()
                .expect("native input plan shape mutex poisoned")
                .clone(),
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
            capability_contract: self
                .state
                .last_capability_contract
                .lock()
                .expect("native capability contract diagnostics mutex poisoned")
                .clone(),
        })
    }
}

impl PhysicalOptimizerRule for NexusNativePhysicalOptimizerRule {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        config: &ConfigOptions,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        self.config.validate().map_err(datafusion::common::DataFusionError::from)?;
        if plan_is_query_admission_wrapped(plan.as_ref()) {
            return Ok(plan);
        }
        let already_selected = if plan.downcast_ref::<NexusNativeExec>().is_some()
            || plan.downcast_ref::<NexusNativeComposedExec>().is_some()
            || is_certified_mixed_execution_plan(plan.as_ref())
        {
            Some(Arc::clone(&plan))
        } else {
            peel_cooperative_native_exec(&plan)
        };
        let retains_prior_selection_report = already_selected.is_some();

        *self.state.last_input_plan_root.lock().expect("native input plan root mutex poisoned") =
            plan.name().to_string();
        *self.state.last_input_plan_shape.lock().expect("native input plan shape mutex poisoned") =
            physical_plan_shape(plan.as_ref());
        *self.state.last_input_join_diagnostics.lock().expect("native input join diagnostics mutex poisoned") =
            physical_join_diagnostics(plan.as_ref());

        let attempt_extension = config.extensions.get::<NexusAttemptExtension>().cloned();
        let effective_config = self.config.clone().with_query_id_if_absent(
            attempt_extension
                .as_ref()
                .and_then(NexusAttemptExtension::query_id)
                .map(|query_id| QueryId::new(query_id.as_str()))
                .unwrap_or_else(QueryId::mint),
        );
        let native_execution_config = effective_config
            .native_execution_config()
            .clone()
            .with_lakehouse_footer_pruning(lakehouse_pruning::enabled_from_env()?);
        let effective_config = effective_config.with_native_execution_config(native_execution_config);
        let native_execution_config = effective_config.native_execution_config();
        let native_optimizer_options = effective_config.native_optimizer_options();
        let native_plan_cost_options = effective_config.native_plan_cost_options();

        let optimized = match (already_selected, self.config.relational_native_rewrite_policy(), self.stage) {
            (Some(plan), _, _) => NativeOptimizationAttempt::retained(plan),
            (None, super::RelationalNativeRewritePolicy::Disabled, _) => {
                NativeOptimizationAttempt::retained(Arc::clone(&plan))
            },
            (None, super::RelationalNativeRewritePolicy::Enabled, NativeOptimizerStage::CountOnlyProbe) => {
                optimize_count_only_native_plan(
                    Arc::clone(&plan),
                    native_execution_config,
                    native_optimizer_options,
                    native_plan_cost_options,
                )?
            },
            (None, super::RelationalNativeRewritePolicy::Enabled, NativeOptimizerStage::Final) => {
                optimize_final_native_plan(
                    Arc::clone(&plan),
                    native_execution_config,
                    native_optimizer_options,
                    native_plan_cost_options,
                )?
            },
        };
        let mut optimized = optimized;
        let binds_final_plan = self.stage == NativeOptimizerStage::Final && self.mints_plan_ticket;
        if binds_final_plan {
            let mut next_occurrence = 0_u64;
            optimized.plan = bind_gpu_function_host_occurrences(optimized.plan, true, &mut next_occurrence)?;
        }
        if binds_final_plan
            && plan_contains_gpu_execution(optimized.plan.as_ref())
            && let Some(extension) = &attempt_extension
        {
            extension.native_admission_planned(optimized.plan.as_ref());
        }
        let query_admission_ticket = if binds_final_plan {
            plan_query_admission_ticket_for_optimized_plan(
                optimized.plan.as_ref(),
                native_execution_config,
                attempt_extension.as_ref(),
            )?
        } else {
            None
        };
        if self.mints_plan_ticket
            && let (Some(extension), Some(ticket)) = (&attempt_extension, &query_admission_ticket)
        {
            extension.bind_ticket(Arc::clone(ticket)).map_err(DataFusionError::from)?;
        }
        if self.mints_plan_ticket {
            apply_execution_admission_wrappers(&mut optimized, query_admission_ticket);
        }
        if !retains_prior_selection_report {
            self.store_optimization_attempt(&optimized)?;
        }
        Ok(optimized.plan)
    }

    fn name(&self) -> &str {
        match self.stage {
            NativeOptimizerStage::CountOnlyProbe => NEXUS_NATIVE_COUNT_ONLY_PHYSICAL_OPTIMIZER_RULE_NAME,
            NativeOptimizerStage::Final => NEXUS_NATIVE_PHYSICAL_OPTIMIZER_RULE_NAME,
        }
    }

    fn schema_check(&self) -> bool {
        true
    }
}

impl NexusNativePhysicalOptimizerRule {
    fn store_optimization_attempt(&self, optimized: &NativeOptimizationAttempt) -> DataFusionResult<()> {
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
        *self.state.last_capability_contract.lock().expect("native capability contract diagnostics mutex poisoned") =
            optimized.capability_contract.clone();
        Ok(())
    }
}

fn apply_execution_admission_wrappers(
    optimized: &mut NativeOptimizationAttempt,
    query_admission_ticket: Option<Arc<QueryAdmissionTicket>>,
) {
    if let Some(ticket) = query_admission_ticket {
        optimized.plan = wrap_plan_with_query_admission_ticket(Arc::clone(&optimized.plan), ticket);
    }
}

/// Mints the rule-equivalent plan admission ticket for a directly constructed
/// native plan and wraps it for execution. Unit tests that build native execs
/// without the optimizer rule use this instead of duplicating ticket plumbing.
#[cfg(test)]
pub(crate) fn wrap_direct_native_plan_with_admission_ticket(
    plan: Arc<dyn ExecutionPlan>,
    native_execution_config: &NexusNativeExecutionConfig,
) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
    let ticket = plan_query_admission_ticket_for_optimized_plan(plan.as_ref(), native_execution_config, None)?;
    Ok(match ticket {
        Some(ticket) => wrap_plan_with_query_admission_ticket(plan, ticket),
        None => plan,
    })
}

fn plan_query_admission_ticket_for_optimized_plan(
    plan: &dyn ExecutionPlan,
    native_execution_config: &NexusNativeExecutionConfig,
    attempt: Option<&NexusAttemptExtension>,
) -> DataFusionResult<Option<Arc<QueryAdmissionTicket>>> {
    let fragment_evidence_items = native_fragment_capability_evidences(plan)?;
    let ticket = native_execution_config
        .new_plan_query_admission_ticket_with_fragment_requirements(fragment_evidence_items)
        .map_err(|error| {
            datafusion::common::DataFusionError::from(Error::from_native(
                "datafusion_nexus.native_optimizer.graph_ticket_ask",
                "failed to compute query admission ticket requirements",
                error,
            ))
        })?;
    let Some(ticket) = ticket else {
        return Ok(None);
    };
    let ticket = match attempt {
        Some(extension) => QueryAdmissionTicket::new_attempt_bound(
            ticket.source(),
            ticket.attempt_capability_evidence_for_rebind(),
            extension.ticket_binding(),
            ticket.metrics_capture(),
        ),
        None => ticket,
    };
    Ok(Some(ticket))
}

fn native_fragment_capability_evidences(plan: &dyn ExecutionPlan) -> DataFusionResult<Vec<AttemptCapabilityEvidence>> {
    let mut evidence_items = Vec::new();
    collect_native_fragment_capability_evidences(plan, false, true, &mut evidence_items)?;
    if let Some(manifest) = gpu_function_execution_manifest(plan)? {
        evidence_items.push(AttemptCapabilityEvidence::empty().with_execution_manifest(manifest).map_err(|error| {
            DataFusionError::from(Error::from_native(
                "datafusion_nexus.native_optimizer.execution_manifest",
                "failed to bind the final GPU function occurrence manifest",
                error,
            ))
        })?);
    }
    Ok(evidence_items)
}

fn gpu_function_execution_manifest(plan: &dyn ExecutionPlan) -> DataFusionResult<Option<AttemptExecutionManifest>> {
    let mut occurrences = Vec::new();
    let _ = collect_gpu_function_occurrences(plan, true, &mut occurrences)?;
    if occurrences.is_empty() {
        return Ok(None);
    }
    AttemptExecutionManifest::new(occurrences).map(Some).map_err(|error| {
        DataFusionError::from(Error::from_native(
            "datafusion_nexus.native_optimizer.execution_manifest",
            "the final GPU function occurrence graph is invalid",
            error,
        ))
    })
}

fn bind_gpu_function_host_occurrences(
    plan: Arc<dyn ExecutionPlan>,
    materializes_host_result: bool,
    next_occurrence: &mut u64,
) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
    if plan.downcast_ref::<GpuFunctionHostOccurrenceExec>().is_some() {
        return Ok(plan);
    }
    if plan.downcast_ref::<NexusNativeComposedExec>().is_some() {
        let children = plan.children().into_iter().cloned().collect::<Vec<_>>();
        let mut rewritten_children = Vec::with_capacity(children.len());
        let mut children_changed = false;
        for child in children {
            let rewritten = bind_gpu_function_host_occurrences(Arc::clone(&child), false, next_occurrence)?;
            children_changed |= !Arc::ptr_eq(&child, &rewritten);
            rewritten_children.push(rewritten);
        }
        return if children_changed {
            plan.replace_children(rewritten_children, ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute))
        } else {
            Ok(plan)
        };
    }
    let producer = crate::native_exec::gpu_resident_producer(plan.as_ref());
    let children = plan.children().into_iter().cloned().collect::<Vec<_>>();
    let mut rewritten_children = Vec::with_capacity(children.len());
    let mut children_changed = false;
    for child in children {
        let child_materializes_host_result =
            producer.is_none() || crate::native_exec::gpu_resident_producer(child.as_ref()).is_none();
        let rewritten =
            bind_gpu_function_host_occurrences(Arc::clone(&child), child_materializes_host_result, next_occurrence)?;
        children_changed |= !Arc::ptr_eq(&child, &rewritten);
        rewritten_children.push(rewritten);
    }
    let plan = if !children_changed {
        plan
    } else {
        plan.replace_children(rewritten_children, ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute))?
    };
    let Some(producer) = crate::native_exec::gpu_resident_producer(plan.as_ref()) else {
        return Ok(plan);
    };
    if !materializes_host_result || producer.host_result_query_plan()?.is_none() {
        return Ok(plan);
    }
    let producer_identity = producer.gpu_result_input_name().to_string();
    let occurrence_id = format!("{producer_identity}#{}", *next_occurrence);
    *next_occurrence = next_occurrence.saturating_add(1);
    Ok(Arc::new(GpuFunctionHostOccurrenceExec::new(plan, occurrence_id, producer_identity)))
}

fn collect_gpu_function_occurrences(
    plan: &dyn ExecutionPlan,
    materializes_host_result: bool,
    occurrences: &mut Vec<AttemptExecutionOccurrence>,
) -> DataFusionResult<Vec<String>> {
    if plan.downcast_ref::<NexusNativeComposedExec>().is_some() {
        let mut predecessors = Vec::new();
        for child in plan.children() {
            for occurrence in collect_gpu_function_occurrences(child.as_ref(), false, occurrences)? {
                if !predecessors.contains(&occurrence) {
                    predecessors.push(occurrence);
                }
            }
        }
        return Ok(predecessors);
    }
    if let Some(bound) = plan.downcast_ref::<GpuFunctionHostOccurrenceExec>() {
        let predecessors = collect_gpu_function_occurrences(bound.input().as_ref(), false, occurrences)?;
        let producer = crate::native_exec::gpu_resident_producer(bound.input().as_ref()).ok_or_else(|| {
            DataFusionError::from(
                Error::new(
                    ErrorCode::InvalidExecutionPlan,
                    "datafusion_nexus.native_optimizer.execution_manifest",
                    "a bound GPU function occurrence no longer wraps a typed producer",
                )
                .with_log_safe_fact("reason", "bound_gpu_occurrence_producer_missing")
                .with_log_safe_fact("occurrence_id", bound.occurrence_id()),
            )
        })?;
        let host_plan = producer.host_result_query_plan()?.ok_or_else(|| {
            DataFusionError::from(
                Error::new(
                    ErrorCode::InvalidExecutionPlan,
                    "datafusion_nexus.native_optimizer.execution_manifest",
                    "a bound GPU function occurrence no longer exposes a host plan",
                )
                .with_log_safe_fact("reason", "bound_gpu_occurrence_host_plan_missing")
                .with_log_safe_fact("occurrence_id", bound.occurrence_id()),
            )
        })?;
        occurrences.push(AttemptExecutionOccurrence::host_arrow_for_producer(
            bound.occurrence_id(),
            bound.producer_identity(),
            &host_plan,
            predecessors,
            producer.host_result_has_deferred_graph_facts(),
        ));
        return Ok(vec![bound.occurrence_id().to_string()]);
    }
    let producer = crate::native_exec::gpu_resident_producer(plan);
    let mut predecessors = Vec::new();
    for child in plan.children() {
        let child_materializes_host_result =
            producer.is_none() || crate::native_exec::gpu_resident_producer(child.as_ref()).is_none();
        for occurrence in collect_gpu_function_occurrences(child.as_ref(), child_materializes_host_result, occurrences)?
        {
            if !predecessors.contains(&occurrence) {
                predecessors.push(occurrence);
            }
        }
    }
    let Some(producer) = producer else {
        return Ok(predecessors);
    };
    if !materializes_host_result {
        return Ok(predecessors);
    }
    if producer.host_result_query_plan()?.is_none() {
        return Ok(predecessors);
    }
    Err(DataFusionError::from(
        Error::new(
            ErrorCode::InvalidExecutionPlan,
            "datafusion_nexus.native_optimizer.execution_manifest",
            "a host-materializing GPU function lacks final-plan occurrence lineage",
        )
        .with_log_safe_fact("reason", "gpu_function_occurrence_lineage_missing")
        .with_log_safe_fact("producer_identity", producer.gpu_result_input_name()),
    ))
}

fn plan_contains_gpu_execution(plan: &dyn ExecutionPlan) -> bool {
    graph_execution_plan(plan).is_some()
        || crate::native_exec::gpu_resident_producer(plan).is_some()
        || lowered_native_plan(plan).is_some()
        || plan.children().iter().any(|child| plan_contains_gpu_execution(child.as_ref()))
}

fn collect_native_fragment_capability_evidences(
    plan: &dyn ExecutionPlan,
    executes_as_cugraph_input: bool,
    materializes_host_result: bool,
    evidence_items: &mut Vec<AttemptCapabilityEvidence>,
) -> DataFusionResult<()> {
    #[cfg(feature = "cugraph")]
    let is_composed = if let Some(composed) = plan.downcast_ref::<NexusNativeComposedExec>() {
        let gpu_plan;
        let composed_plan = if executes_as_cugraph_input {
            gpu_plan = composed.native_plan().clone_with_gpu_resident_sink();
            &gpu_plan
        } else {
            composed.native_plan()
        };
        evidence_items
            .push(crate::native_exec::composed_admission_capability_evidence(composed_plan, composed.gpu_inputs())?);
        true
    } else {
        false
    };
    #[cfg(not(feature = "cugraph"))]
    let is_composed = false;
    if is_composed {
        for child in plan.children() {
            let child_is_cugraph = graph_execution_plan(child.as_ref()).is_some();
            for descendant in child.children() {
                collect_native_fragment_capability_evidences(
                    descendant.as_ref(),
                    executes_as_cugraph_input || child_is_cugraph,
                    false,
                    evidence_items,
                )?;
            }
        }
        return Ok(());
    }
    let graph_execution_evidence =
        graph_execution_plan(plan).map(GraphExecutionPlan::graph_execution_capability_evidence).transpose()?;
    let is_cugraph_execution = graph_execution_evidence.is_some();
    let producer = crate::native_exec::gpu_resident_producer(plan);
    for child in plan.children() {
        let child_materializes_host_result =
            producer.is_none() || crate::native_exec::gpu_resident_producer(child.as_ref()).is_none();
        collect_native_fragment_capability_evidences(
            child.as_ref(),
            executes_as_cugraph_input || is_cugraph_execution,
            child_materializes_host_result,
            evidence_items,
        )?;
    }
    if let Some(evidence) = graph_execution_evidence {
        evidence_items.push(evidence);
        return Ok(());
    }
    if let Some(producer) = producer {
        evidence_items.push(if materializes_host_result {
            producer.host_result_capability_evidence()?
        } else {
            producer.gpu_result_capability_evidence()?
        });
        return Ok(());
    }
    if let Some(native_plan) = lowered_native_plan(plan) {
        let gpu_resident_plan;
        let native_plan = if executes_as_cugraph_input {
            gpu_resident_plan = native_plan.clone_with_gpu_resident_sink();
            &gpu_resident_plan
        } else {
            native_plan
        };
        evidence_items.push(native_plan_capability_evidence(native_plan)?);
    }
    Ok(())
}

fn native_plan_capability_evidence(
    native_plan: &nexus_query_engine::plan::QueryPlan,
) -> DataFusionResult<AttemptCapabilityEvidence> {
    Ok(AttemptCapabilityEvidence::for_query_plan(native_plan))
}

fn lowered_native_plan(plan: &dyn ExecutionPlan) -> Option<&nexus_query_engine::plan::QueryPlan> {
    if let Some(native) = plan.downcast_ref::<NexusNativeExec>() {
        return Some(native.native_plan());
    }
    if let Some(native) = plan.downcast_ref::<NexusNativeComposedExec>() {
        return Some(native.native_plan());
    }
    if let Some(native) = plan.downcast_ref::<NexusNativeGpuProducerExec>() {
        return Some(native.native_plan());
    }
    plan.downcast_ref::<NexusNativeComposedGpuProducerExec>().map(NexusNativeComposedGpuProducerExec::native_plan)
}

#[cfg(test)]
mod tests {
    use arrow::datatypes::{DataType, Field, Schema};
    #[cfg(feature = "cugraph")]
    use datafusion::physical_plan::coop::CooperativeExec;
    use datafusion::physical_plan::empty::EmptyExec;
    use nexus_query_engine::{
        plan::{QueryPlan, QueryPlanBuilder},
        source::{GpuResidentSource, LocalParquetFile, LocalParquetSource, SourceNode},
    };

    use super::*;
    #[cfg(feature = "cugraph")]
    use crate::cugraph_sql::{
        CugraphAlgoExec, CugraphAlgorithm, CugraphAlgorithmParams, CugraphAlgorithmSpec,
        CugraphGraphConstructionPolicy, CugraphGraphOptions, CugraphTableReference, CugraphTriangleCountAllParams,
        result_schema_for_params,
    };
    use crate::{
        backend::InstalledTestGpuBackend,
        error::{DataFusionErrorExt, ErrorCode, ErrorStatus},
    };

    fn local_parquet_native_plan(path: &str) -> QueryPlan {
        let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
        let file = LocalParquetFile::new(path)
            .with_file_size_bytes(8)
            .with_row_group_bytes(vec![8])
            .with_projected_row_group_bytes(vec![8], vec![8])
            .with_row_group_rows(vec![1]);
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
            vec![file],
            schema,
            vec!["value".to_string()],
        )));
        builder.finish_with_host_sink(source)
    }

    fn native_exec_leaf() -> Arc<dyn ExecutionPlan> {
        Arc::new(
            NexusNativeExec::try_new(
                local_parquet_native_plan("/tmp/datafusion-nexus-final-plan-classifier-not-read.parquet"),
                NexusNativeExecutionConfig::bounded_default(),
            )
            .expect("native runtime leaf should build without opening its source"),
        )
    }

    #[test]
    fn final_plan_classifier_distinguishes_datafusion_and_native_runtime_nodes() {
        let datafusion: Arc<dyn ExecutionPlan> = Arc::new(EmptyExec::new(Arc::new(Schema::empty())));

        assert_eq!(classify_final_plan(datafusion.as_ref()), FinalPlanDisposition::DataFusion);
        assert_eq!(classify_final_plan(native_exec_leaf().as_ref()), FinalPlanDisposition::Native);
    }

    #[test]
    fn reject_datafusion_cpu_policy_rejects_the_exact_datafusion_final_plan() {
        let plan: Arc<dyn ExecutionPlan> = Arc::new(EmptyExec::new(Arc::new(Schema::empty())));
        let rule = NexusFinalPlanRequirementRule::new(
            NexusNativeOptimizerConfig::default()
                .with_datafusion_cpu_execution_policy(DataFusionCpuExecutionPolicy::Reject),
        );

        let error = rule
            .optimize(plan, &ConfigOptions::default())
            .expect_err("the reject policy must reject a CPU-only final plan");
        let error = error.find_nexus_error().expect("typed adapter error");

        assert_eq!(error.code(), ErrorCode::FinalPlanRequirementUnsatisfied);
        assert_eq!(error.status(), ErrorStatus::Permanent);
        assert_eq!(error.fact_value("requirement"), Some("no_datafusion_cpu"));
        assert_eq!(error.fact_value("observed_final_plan_disposition"), Some("datafusion"));
        assert_eq!(error.fact_value("terminal_plan_disposition"), Some("rejected"));
    }

    #[test]
    fn strict_rejection_schema_detail_is_client_visible_but_not_log_safe() {
        use crate::{
            capability::CapabilityReason,
            report::{GpuCandidate, UnsupportedColumnDetail},
        };

        let mut report = PlanningReport::new();
        report.push_not_supported_column(
            GpuCandidate::new("ParquetExec", 1),
            CapabilityReason::UnsupportedType,
            UnsupportedColumnDetail::new("訂單明細", "List(Int64)", "projected"),
        );
        let plan: Arc<dyn ExecutionPlan> = Arc::new(EmptyExec::new(Arc::new(Schema::empty())));
        let error = final_plan_requirement_error(plan.as_ref(), FinalPlanDisposition::DataFusion, Some(&report));

        assert_eq!(error.code(), ErrorCode::FinalPlanRequirementUnsatisfied);
        assert!(error.message().contains("訂單明細"), "{}", error.message());
        assert_eq!(error.fact_value("unsupported_column"), Some("訂單明細"));
        assert_eq!(error.fact_value("unsupported_column_type"), Some("List(Int64)"));
        assert_eq!(error.fact_value("unsupported_column_role"), Some("projected"));
        let logs = error.facts().log_safe_projection();
        assert!(
            !logs.rendered().unwrap_or_default().contains("訂單明細"),
            "operational logs must not carry caller schema identifiers"
        );
        assert!(
            error
                .facts()
                .retained_capture_entries()
                .contains(&("unsupported_column".to_string(), "訂單明細".to_string())),
            "query-caller capture must retain the strict-rejection detail"
        );
    }

    #[test]
    fn allow_datafusion_cpu_policy_returns_the_original_plan_without_replanning() {
        let plan: Arc<dyn ExecutionPlan> = Arc::new(EmptyExec::new(Arc::new(Schema::empty())));
        let rule = NexusFinalPlanRequirementRule::new(NexusNativeOptimizerConfig::default());

        let returned = rule.optimize(Arc::clone(&plan), &ConfigOptions::default()).unwrap();

        assert!(Arc::ptr_eq(&plan, &returned));
    }

    fn backend_admission_config(min_bytes: u64) -> (InstalledTestGpuBackend, NexusNativeExecutionConfig) {
        let backend = InstalledTestGpuBackend::new();
        let source = backend
            .optimizer_config()
            .native_execution_config()
            .query_admission_source()
            .expect("installed backend must provide admission authority");
        let config = NexusNativeExecutionConfig::bounded(min_bytes, 512, 1, 3, 1).with_query_admission_source(source);
        (backend, config)
    }

    #[cfg(feature = "cugraph")]
    fn edge_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("src", DataType::Int64, false),
            Field::new("dst", DataType::Int64, false),
            Field::new("weight", DataType::Float64, false),
        ]))
    }

    #[cfg(feature = "cugraph")]
    fn deferred_cugraph_exec(config: &NexusNativeExecutionConfig) -> Arc<dyn ExecutionPlan> {
        let params = CugraphAlgorithmParams::TriangleCountAll(CugraphTriangleCountAllParams);
        let spec = CugraphAlgorithmSpec {
            algorithm: CugraphAlgorithm::TriangleCountAll,
            table_name: CugraphTableReference::parse("edges").expect("fixture table name is valid"),
            src_col: "src".to_string(),
            dst_col: "dst".to_string(),
            weight_col: Some("weight".to_string()),
            edge_id_col: None,
            graph_options: CugraphGraphOptions {
                directed: false,
                renumber: true,
                construction_policy: CugraphGraphConstructionPolicy::PythonCugraph,
            },
            output_schema: result_schema_for_params(CugraphAlgorithm::TriangleCountAll, &params),
            params,
        };
        let edges = Arc::new(EmptyExec::new(edge_schema())) as Arc<dyn ExecutionPlan>;
        Arc::new(
            CugraphAlgoExec::try_new(spec, edges, vec![], None, None, config.clone())
                .expect("deferred cuGraph fixture should build"),
        ) as Arc<dyn ExecutionPlan>
    }

    #[test]
    fn planner_ticket_aggregates_scopes_from_composed_native_fragments() {
        let (_backend, config) = backend_admission_config(64 * 1024 * 1024);
        let left = NexusNativeGpuProducerExec::try_new(
            "left",
            local_parquet_native_plan("/tmp/datafusion-nexus-planner-left-not-read.parquet")
                .clone_with_gpu_resident_sink(),
        )
        .unwrap();
        let right = NexusNativeGpuProducerExec::try_new(
            "right",
            local_parquet_native_plan("/tmp/datafusion-nexus-planner-right-not-read.parquet")
                .clone_with_gpu_resident_sink(),
        )
        .unwrap();
        let schema = left.schema();
        let mut builder = QueryPlanBuilder::new();
        let left_source =
            builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("left", Arc::clone(&schema))));
        let right_source =
            builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("right", Arc::clone(&schema))));
        let union = builder.add_union(vec![left_source, right_source], Arc::clone(&schema));
        let consumer_plan = builder.finish_with_host_sink_schema(union, schema);

        let expected = crate::native_exec::composed_admission_capability_evidence(
            &consumer_plan,
            &[Arc::new(left.clone()), Arc::new(right.clone())],
        )
        .expect("composed evidence binds both producer occurrences");
        let composed =
            NexusNativeComposedExec::try_new(consumer_plan, vec![Arc::new(left), Arc::new(right)], config.clone())
                .unwrap();

        let ticket = plan_query_admission_ticket_for_optimized_plan(&composed, &config, None).unwrap().unwrap();

        assert_eq!(ticket.attempt_capability_evidence(), Some(expected));
        let evidence = ticket.attempt_capability_evidence().expect("planner ticket retains its evidence");
        assert!(evidence.required_capabilities().is_empty());
        assert_eq!(evidence.execution_scopes().len(), 3);
    }

    #[cfg(feature = "cugraph")]
    #[test]
    fn planner_binds_outer_ticket_for_registered_graph_descendant() {
        let (_backend, config) = backend_admission_config(64 * 1024 * 1024);
        let graph = deferred_cugraph_exec(&config);
        let plan = Arc::new(CooperativeExec::new(graph)) as Arc<dyn ExecutionPlan>;
        let mut next_occurrence = 0;
        let plan = bind_gpu_function_host_occurrences(plan, true, &mut next_occurrence)
            .expect("final-plan binding should assign graph occurrence lineage");

        let ticket = plan_query_admission_ticket_for_optimized_plan(plan.as_ref(), &config, None)
            .expect("registered cuGraph discovery should not fail")
            .expect("the final plan must own one ticket before deferred graph preparation");
        let evidence = ticket.attempt_capability_evidence().expect("graph ticket retains initial evidence");
        assert_eq!(evidence.execution_manifest().unwrap().occurrences().len(), 1);
    }

    #[test]
    fn planner_keeps_outer_ticket_for_non_graph_native_plan() {
        let (_backend, config) = backend_admission_config(64 * 1024 * 1024);
        let plan = native_exec_leaf();

        let ticket = plan_query_admission_ticket_for_optimized_plan(plan.as_ref(), &config, None)
            .expect("non-graph native ticket planning should succeed")
            .expect("non-graph native plan must retain normal outer admission");
        let wrapped = wrap_plan_with_query_admission_ticket(plan, ticket);

        assert!(
            plan_is_query_admission_wrapped(wrapped.as_ref()),
            "deferring cuGraph admission must not suppress ordinary native admission wrappers",
        );
    }

    #[test]
    fn final_rule_binds_a_ticket_to_a_plan_selected_by_the_count_probe() {
        let (backend, config) = backend_admission_config(64 * 1024 * 1024);
        let optimizer_config = backend.optimizer_config().clone().with_native_execution_config(config);
        let rule = NexusNativePhysicalOptimizerRule::new(optimizer_config);
        let selected_by_count_probe = native_exec_leaf();

        let final_plan = rule.optimize(selected_by_count_probe, &ConfigOptions::default()).unwrap();

        assert!(
            plan_is_query_admission_wrapped(final_plan.as_ref()),
            "the final pass must bind admission even when relational selection already happened"
        );
    }
}
