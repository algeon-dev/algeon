use std::sync::Arc;

use datafusion::{
    common::{DataFusionError, Result as DataFusionResult, config::ConfigOptions},
    execution::SessionState,
    logical_expr::LogicalPlan,
    physical_optimizer::PhysicalOptimizerRule,
    physical_plan::{ChildrenPropertiesMode, ExecutionPlan, ReplaceChildrenOptions},
};
use query_engine::{
    admission::{AttemptCapabilityEvidence, AttemptExecutionManifest, AttemptExecutionOccurrence},
    metrics::QueryId,
    optimizer::NativeOptimizerDiagnostics,
    plan::QueryPlan,
    source::SourceDiagnostics,
};

use super::{AlgeonNativeOptimizerConfig, DataFusionCpuExecutionPolicy, FinalPlanDisposition, classify_final_plan};
#[cfg(feature = "cugraph")]
use crate::native_exec::composed_admission_capability_evidence;
use crate::{
    attempt::InvocationAttempt,
    capability::CapabilityReason,
    cugraph_sql::is_feature_disabled_execution_plan,
    error::{Error, ErrorCode},
    native::lakehouse_pruning,
    native_exec::{
        AlgeonNativeComposedExec, AlgeonNativeComposedGpuProducerExec, AlgeonNativeExec, AlgeonNativeExecutionConfig,
        AlgeonNativeGpuProducerExec, GpuFunctionHostOccurrenceExec,
        admission::QueryAdmissionTicket,
        gpu_resident_producer,
        graph::{GraphExecutionPlan, graph_execution_plan},
        plan_is_query_admission_wrapped, wrap_plan_with_query_admission_ticket,
    },
    native_report::{NativeJoinDiagnosticSummary, physical_join_diagnostics},
    physical_plan::physical_plan_shape,
    report::{CandidateOutcome, GpuCandidate, PlanningReport, UnsupportedColumnDetail},
    session::{ALGEON_NATIVE_COUNT_ONLY_PHYSICAL_OPTIMIZER_RULE_NAME, ALGEON_NATIVE_PHYSICAL_OPTIMIZER_RULE_NAME},
};

mod attempt;
mod capability_envelope;
mod invocation;
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
pub(crate) use invocation::Invocation;
#[cfg(feature = "server")]
pub(crate) use invocation::InvocationExecution;
use mixed::is_certified_mixed_execution_plan;

/// Rejects a completed plan that executes DataFusion CPU operators.
///
/// `report` is the planning report of the rewrite that produced `plan`, used
/// only to name the first unsupported candidate in the error.
fn enforce_no_datafusion_cpu(plan: &dyn ExecutionPlan, report: Option<&PlanningReport>) -> DataFusionResult<()> {
    let disposition = classify_final_plan(plan);
    // A feature-disabled cuGraph plan is a zero-child terminal capability
    // rejection. It executes neither a CPU fallback nor a GPU operator, so it
    // satisfies the no-CPU-execution contract without being misclassified as
    // a native workload.
    if disposition != FinalPlanDisposition::Native && !is_feature_disabled_execution_plan(plan) {
        return Err(final_plan_requirement_error(plan, disposition, report).into());
    }
    Ok(())
}

pub(crate) fn final_plan_requirement_error(
    plan: &dyn ExecutionPlan,
    disposition: FinalPlanDisposition,
    report: Option<&PlanningReport>,
) -> Error {
    let rejection = report.and_then(first_native_not_supported);
    let message = rejection.and_then(|(_, _, detail)| detail).map_or_else(
        || "exact final physical plan does not satisfy the configured GPU execution requirement".to_string(),
        |detail| format!("GPU execution does not support a {} column with an unsupported Arrow type", detail.role),
    );
    let mut error = Error::new(
        ErrorCode::FinalPlanRequirementUnsatisfied,
        "algeon_datafusion.planner.final_plan_requirement",
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
            // from operational logs and Debug by default, and out of
            // `message()`, which the server logs verbatim. The server
            // projection owner retains them explicitly for query-caller
            // surfaces (Flight metadata allowlist, query-stats capture) and
            // composes them into the actionable status message.
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
) -> Option<(&GpuCandidate, CapabilityReason, Option<&UnsupportedColumnDetail>)> {
    report.outcomes().iter().enumerate().find_map(|(index, outcome)| match outcome {
        CandidateOutcome::NotSupported { candidate, capability_reason } => {
            Some((candidate, *capability_reason, report.unsupported_column(index)))
        },
        CandidateOutcome::Selected { .. }
        | CandidateOutcome::Deferred { .. }
        | CandidateOutcome::NotSelectedByCost { .. } => None,
    })
}

/// Native planning diagnostics of one optimizer stage run.
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
    pub native_optimizer: NativeOptimizerDiagnostics,
    /// Source-read diagnostics captured while lowering/planning native sources.
    pub source_diagnostics: Vec<SourceDiagnostics>,
    /// Canonical capability-contract evidence.
    pub capability_contract: NativeCapabilityContractDiagnostics,
}

/// Plans `logical` once on `state` and returns the native diagnostics of this
/// call only.
///
/// The call plans under its own invocation, detached from any attempt that
/// `state` belongs to, so it binds no attempt admission. The diagnostics
/// describe the latest native optimizer stage of this planning pass. A session
/// without the native optimizer rule records none and fails with
/// `reason = "native_optimizer_rule_not_installed"`.
pub async fn plan_with_native_diagnostics(
    state: &SessionState,
    logical: &LogicalPlan,
) -> DataFusionResult<(Arc<dyn ExecutionPlan>, NativePlanningDiagnostics)> {
    let invocation = Invocation::detached();
    let mut state = state.clone();
    invocation.install(state.config_mut());
    let plan = state.create_physical_plan(logical).await?;
    let diagnostics = invocation.planning_diagnostics().ok_or_else(|| {
        DataFusionError::from(
            Error::new(
                ErrorCode::InvalidRequest,
                "algeon_datafusion.planner.native_diagnostics",
                "the session has no native optimizer rule to record planning diagnostics",
            )
            .with_log_safe_fact("reason", "native_optimizer_rule_not_installed"),
        )
    })?;
    Ok((plan, diagnostics))
}

/// DataFusion physical optimizer rule that rewrites supported plans to native cuDF execution.
///
/// The rule keeps no per-query state. Planning diagnostics belong to the
/// invocation that planned and are read through
/// [`AlgeonAttempt::planning_diagnostics`](crate::attempt::AlgeonAttempt::planning_diagnostics),
/// [`plan_with_native_diagnostics`], or [`Self::optimize_with_diagnostics`].
/// Session insertion deduplicates by the public rule names in [`crate::session`].
/// Candidate misses leave the original DataFusion plan intact. Under a reject
/// DataFusion CPU policy, the final stage rejects the plan it completed.
#[derive(Debug)]
pub struct AlgeonNativePhysicalOptimizerRule {
    config: AlgeonNativeOptimizerConfig,
    stage: NativeOptimizerStage,
    // False for a cuGraph input subplan: that plan is a child of the query
    // root, so the outer run binds admission and checks the CPU policy.
    plans_query_root: bool,
}

// Distinct rule identities let the count-only probe run before the final rewrite pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NativeOptimizerStage {
    // Dedicated pass for plans where output-requirement peeling exposes a native count.
    CountOnlyProbe,
    // Full native lowering, fallback handling, and fragment composition.
    Final,
}

impl AlgeonNativePhysicalOptimizerRule {
    /// Create the final native rewrite rule.
    pub fn new(config: AlgeonNativeOptimizerConfig) -> Self {
        Self { config, stage: NativeOptimizerStage::Final, plans_query_root: true }
    }

    #[cfg(feature = "cugraph")]
    pub(crate) fn new_cugraph_subplan(config: AlgeonNativeOptimizerConfig) -> Self {
        Self { config, stage: NativeOptimizerStage::Final, plans_query_root: false }
    }

    /// Return the immutable config copied into rewritten native exec nodes.
    pub fn config(&self) -> &AlgeonNativeOptimizerConfig {
        &self.config
    }

    pub(crate) fn count_only_probe_rule(&self) -> Self {
        Self {
            config: self.config.clone(),
            stage: NativeOptimizerStage::CountOnlyProbe,
            plans_query_root: self.plans_query_root,
        }
    }

    /// Runs this rule once over `plan` and returns the result with this
    /// call's planning diagnostics.
    ///
    /// The call plans under its own invocation, detached from any attempt
    /// carried by `options`. The diagnostics are `None` when `plan` already
    /// holds this rule's native selection, which the rule keeps unchanged.
    pub fn optimize_with_diagnostics(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        options: &ConfigOptions,
    ) -> DataFusionResult<(Arc<dyn ExecutionPlan>, Option<NativePlanningDiagnostics>)> {
        let invocation = Invocation::detached();
        let mut options = options.clone();
        options.extensions.insert(invocation.clone());
        let plan = self.optimize(plan, &options)?;
        Ok((plan, invocation.planning_diagnostics()))
    }

    fn ends_planning_pass(&self) -> bool {
        self.stage == NativeOptimizerStage::Final && self.plans_query_root
    }
}

impl PhysicalOptimizerRule for AlgeonNativePhysicalOptimizerRule {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        config: &ConfigOptions,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let invocation = Invocation::of_options(config);
        let rewritten = {
            let _footers = invocation.map(Invocation::footer_scope);
            self.rewrite(plan, invocation)
        };
        if self.ends_planning_pass()
            && let Some(invocation) = invocation
        {
            invocation.end_planning_pass();
        }
        let (plan, report) = rewritten?;
        // The last step of the final stage, on the plan and report of this
        // rewrite. `crate::session` installs this stage after every other rule
        // already in the pipeline and immediately before DataFusion's sanity
        // check.
        if self.ends_planning_pass()
            && self.config.datafusion_cpu_execution_policy() == DataFusionCpuExecutionPolicy::Reject
        {
            // A plan this rule already selected recorded its report in the
            // earlier stage that selected it.
            let prior = || invocation.and_then(Invocation::planning_diagnostics).map(|diagnostics| diagnostics.report);
            enforce_no_datafusion_cpu(plan.as_ref(), report.or_else(prior).as_ref())?;
        }
        Ok(plan)
    }

    fn name(&self) -> &str {
        match self.stage {
            NativeOptimizerStage::CountOnlyProbe => ALGEON_NATIVE_COUNT_ONLY_PHYSICAL_OPTIMIZER_RULE_NAME,
            NativeOptimizerStage::Final => ALGEON_NATIVE_PHYSICAL_OPTIMIZER_RULE_NAME,
        }
    }

    fn schema_check(&self) -> bool {
        true
    }
}

impl AlgeonNativePhysicalOptimizerRule {
    /// Rewrites `plan` and returns it with this rewrite's planning report, or
    /// `None` when `plan` already held a native selection.
    fn rewrite(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        invocation: Option<&Invocation>,
    ) -> DataFusionResult<(Arc<dyn ExecutionPlan>, Option<PlanningReport>)> {
        self.config.validate().map_err(DataFusionError::from)?;
        if plan_is_query_admission_wrapped(plan.as_ref()) {
            return Ok((plan, None));
        }
        let already_selected = if plan.downcast_ref::<AlgeonNativeExec>().is_some()
            || plan.downcast_ref::<AlgeonNativeComposedExec>().is_some()
            || is_certified_mixed_execution_plan(plan.as_ref())
        {
            Some(Arc::clone(&plan))
        } else {
            peel_cooperative_native_exec(&plan)
        };
        let retains_prior_selection_report = already_selected.is_some();
        // The DataFusion input as this rewrite saw it; native replacement can
        // remove the join nodes these diagnostics describe.
        let input = invocation.filter(|_| !retains_prior_selection_report).map(|_| {
            (plan.name().to_string(), physical_plan_shape(plan.as_ref()), physical_join_diagnostics(plan.as_ref()))
        });

        let attempt = invocation.and_then(Invocation::attempt);
        let native_execution_config = self.config.native_execution_config().clone();
        let native_execution_config = match invocation {
            Some(invocation) => invocation.execution_config(native_execution_config),
            None => native_execution_config,
        }
        .with_query_id_if_absent(
            attempt
                .and_then(InvocationAttempt::query_id)
                .map(|query_id| QueryId::new(query_id.as_str()))
                .unwrap_or_else(QueryId::mint),
        )
        .with_lakehouse_footer_pruning(lakehouse_pruning::enabled_from_env()?);
        let effective_config = self.config.clone().with_native_execution_config(native_execution_config);
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
        let binds_final_plan = self.ends_planning_pass();
        if binds_final_plan {
            let mut next_occurrence = 0_u64;
            optimized.plan = bind_gpu_function_host_occurrences(optimized.plan, true, &mut next_occurrence)?;
        }
        if binds_final_plan
            && plan_contains_gpu_execution(optimized.plan.as_ref())
            && let Some(attempt) = attempt
        {
            attempt.native_admission_planned(optimized.plan.as_ref());
        }
        let query_admission_ticket = if binds_final_plan {
            plan_query_admission_ticket_for_optimized_plan(
                optimized.plan.as_ref(),
                native_execution_config,
                invocation,
            )?
        } else {
            None
        };
        if self.plans_query_root {
            apply_execution_admission_wrappers(&mut optimized, query_admission_ticket);
        }
        if retains_prior_selection_report {
            return Ok((optimized.plan, None));
        }
        if let (Some(invocation), Some((input_plan_root, input_plan_shape, input_join_diagnostics))) =
            (invocation, input)
        {
            invocation.record_planning(NativePlanningDiagnostics {
                input_plan_root,
                input_plan_shape,
                input_join_diagnostics,
                report: optimized.report.clone(),
                native_optimizer: optimized.native_optimizer,
                source_diagnostics: optimized.source_diagnostics,
                capability_contract: optimized.capability_contract,
            });
        }
        Ok((optimized.plan, Some(optimized.report)))
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
    native_execution_config: &AlgeonNativeExecutionConfig,
) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
    let ticket = plan_query_admission_ticket_for_optimized_plan(plan.as_ref(), native_execution_config, None)?;
    Ok(match ticket {
        Some(ticket) => wrap_plan_with_query_admission_ticket(plan, ticket),
        None => plan,
    })
}

/// The admission ticket for a final plan. An attempt invocation mints its one
/// ticket from the plan's evidence, or extends a ticket it already minted;
/// every other plan gets its own plan ticket.
fn plan_query_admission_ticket_for_optimized_plan(
    plan: &dyn ExecutionPlan,
    native_execution_config: &AlgeonNativeExecutionConfig,
    invocation: Option<&Invocation>,
) -> DataFusionResult<Option<Arc<QueryAdmissionTicket>>> {
    let ticket_ask_error = |error| {
        DataFusionError::from(Error::from_native(
            "algeon_datafusion.native_optimizer.graph_ticket_ask",
            "failed to compute query admission ticket requirements",
            error,
        ))
    };
    let fragment_evidence_items = native_fragment_capability_evidences(plan)?;
    let attempt_ticket = match (invocation, native_execution_config.query_admission_source()) {
        (Some(invocation), Some(source)) if invocation.attempt().is_some() && !fragment_evidence_items.is_empty() => {
            let evidence =
                AttemptCapabilityEvidence::aggregate(fragment_evidence_items.clone()).map_err(ticket_ask_error)?;
            invocation.attempt_admission_ticket(source, Some(evidence))
        },
        _ => None,
    };
    match attempt_ticket {
        Some((ticket, true)) => Ok(Some(ticket)),
        Some((ticket, false)) => {
            ticket.extend_capability_evidence_before_admission(fragment_evidence_items).map_err(ticket_ask_error)?;
            Ok(Some(ticket))
        },
        None => native_execution_config
            .new_plan_query_admission_ticket_with_fragment_requirements(fragment_evidence_items)
            .map_err(ticket_ask_error),
    }
}

fn native_fragment_capability_evidences(plan: &dyn ExecutionPlan) -> DataFusionResult<Vec<AttemptCapabilityEvidence>> {
    let mut evidence_items = Vec::new();
    collect_native_fragment_capability_evidences(plan, false, true, &mut evidence_items)?;
    if let Some(manifest) = gpu_function_execution_manifest(plan)? {
        evidence_items.push(AttemptCapabilityEvidence::empty().with_execution_manifest(manifest).map_err(|error| {
            DataFusionError::from(Error::from_native(
                "algeon_datafusion.native_optimizer.execution_manifest",
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
            "algeon_datafusion.native_optimizer.execution_manifest",
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
    if plan.downcast_ref::<AlgeonNativeComposedExec>().is_some() {
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
    let producer = gpu_resident_producer(plan.as_ref());
    let children = plan.children().into_iter().cloned().collect::<Vec<_>>();
    let mut rewritten_children = Vec::with_capacity(children.len());
    let mut children_changed = false;
    for child in children {
        let child_materializes_host_result = producer.is_none() || gpu_resident_producer(child.as_ref()).is_none();
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
    let Some(producer) = gpu_resident_producer(plan.as_ref()) else {
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
    if plan.downcast_ref::<AlgeonNativeComposedExec>().is_some() {
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
        let producer = gpu_resident_producer(bound.input().as_ref()).ok_or_else(|| {
            DataFusionError::from(
                Error::new(
                    ErrorCode::InvalidExecutionPlan,
                    "algeon_datafusion.native_optimizer.execution_manifest",
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
                    "algeon_datafusion.native_optimizer.execution_manifest",
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
    let producer = gpu_resident_producer(plan);
    let mut predecessors = Vec::new();
    for child in plan.children() {
        let child_materializes_host_result = producer.is_none() || gpu_resident_producer(child.as_ref()).is_none();
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
            "algeon_datafusion.native_optimizer.execution_manifest",
            "a host-materializing GPU function lacks final-plan occurrence lineage",
        )
        .with_log_safe_fact("reason", "gpu_function_occurrence_lineage_missing")
        .with_log_safe_fact("producer_identity", producer.gpu_result_input_name()),
    ))
}

fn plan_contains_gpu_execution(plan: &dyn ExecutionPlan) -> bool {
    graph_execution_plan(plan).is_some()
        || gpu_resident_producer(plan).is_some()
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
    let is_composed = if let Some(composed) = plan.downcast_ref::<AlgeonNativeComposedExec>() {
        let gpu_plan;
        let composed_plan = if executes_as_cugraph_input {
            gpu_plan = composed.native_plan().clone_with_gpu_resident_sink();
            &gpu_plan
        } else {
            composed.native_plan()
        };
        evidence_items.push(composed_admission_capability_evidence(composed_plan, composed.gpu_inputs())?);
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
    let producer = gpu_resident_producer(plan);
    for child in plan.children() {
        let child_materializes_host_result = producer.is_none() || gpu_resident_producer(child.as_ref()).is_none();
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

fn native_plan_capability_evidence(native_plan: &QueryPlan) -> DataFusionResult<AttemptCapabilityEvidence> {
    Ok(AttemptCapabilityEvidence::for_query_plan(native_plan))
}

fn lowered_native_plan(plan: &dyn ExecutionPlan) -> Option<&QueryPlan> {
    if let Some(native) = plan.downcast_ref::<AlgeonNativeExec>() {
        return Some(native.native_plan());
    }
    if let Some(native) = plan.downcast_ref::<AlgeonNativeComposedExec>() {
        return Some(native.native_plan());
    }
    if let Some(native) = plan.downcast_ref::<AlgeonNativeGpuProducerExec>() {
        return Some(native.native_plan());
    }
    plan.downcast_ref::<AlgeonNativeComposedGpuProducerExec>().map(AlgeonNativeComposedGpuProducerExec::native_plan)
}

#[cfg(test)]
mod tests {
    use arrow::datatypes::{DataType, Field, Fields, Schema, SchemaRef};
    #[cfg(feature = "cugraph")]
    use datafusion::physical_plan::coop::CooperativeExec;
    use datafusion::physical_plan::{empty::EmptyExec, union::UnionExec};
    #[cfg(feature = "cugraph")]
    use query_engine::source::GpuResidentSource;
    use query_engine::{
        plan::{QueryPlan, QueryPlanBuilder},
        source::{LocalParquetFile, LocalParquetSource, SourceNode},
    };

    use super::*;
    #[cfg(feature = "cugraph")]
    use crate::cugraph_sql::{
        CugraphAlgoExec, CugraphAlgorithm, CugraphAlgorithmParams, CugraphAlgorithmSpec,
        CugraphGraphConstructionPolicy, CugraphGraphOptions, CugraphTriangleCountAllParams, result_schema_for_params,
    };
    use crate::{
        backend::InstalledTestGpuBackend,
        error::{DataFusionErrorExt, ErrorCode, ErrorStatus},
        report::NotSupportedDetail,
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
            AlgeonNativeExec::try_new(
                local_parquet_native_plan("/tmp/algeon-datafusion-final-plan-classifier-not-read.parquet"),
                AlgeonNativeExecutionConfig::bounded_default(),
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
        let plan = union_over_empty_with_unsupported_field("unsupported");
        let rule = AlgeonNativePhysicalOptimizerRule::new(
            AlgeonNativeOptimizerConfig::default()
                .with_datafusion_cpu_execution_policy(DataFusionCpuExecutionPolicy::Reject),
        );

        let error = rule
            .optimize(plan, &ConfigOptions::default())
            .expect_err("the reject policy must reject a CPU-only final plan");
        let error = error.find_algeon_error().expect("typed adapter error");

        assert_eq!(error.code(), ErrorCode::FinalPlanRequirementUnsatisfied);
        assert_eq!(error.status(), ErrorStatus::Permanent);
        assert_eq!(error.fact_value("requirement"), Some("no_datafusion_cpu"));
        assert_eq!(error.fact_value("observed_final_plan_disposition"), Some("datafusion"));
        assert_eq!(error.fact_value("terminal_plan_disposition"), Some("rejected"));
    }

    #[test]
    fn strict_rejection_schema_detail_is_client_visible_but_not_log_safe() {
        let mut report = PlanningReport::new();
        report.push_not_supported_with_detail(
            GpuCandidate::new("ParquetExec", 1),
            CapabilityReason::UnsupportedType,
            NotSupportedDetail {
                column: Some(UnsupportedColumnDetail::new("訂單明細", "List(Int64)", "projected")),
                expression: None,
            },
        );
        let plan: Arc<dyn ExecutionPlan> = Arc::new(EmptyExec::new(Arc::new(Schema::empty())));
        let error = final_plan_requirement_error(plan.as_ref(), FinalPlanDisposition::DataFusion, Some(&report));

        assert_eq!(error.code(), ErrorCode::FinalPlanRequirementUnsatisfied);
        assert!(!error.message().contains("訂單明細"), "{}", error.message());
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

    fn backend_admission_config(min_bytes: u64) -> (InstalledTestGpuBackend, AlgeonNativeExecutionConfig) {
        let backend = InstalledTestGpuBackend::new();
        let source = backend
            .optimizer_config()
            .native_execution_config()
            .query_admission_source()
            .expect("installed backend must provide admission authority");
        let config = AlgeonNativeExecutionConfig::bounded(min_bytes, 512, 1, 3, 1).with_query_admission_source(source);
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
    fn deferred_cugraph_exec(config: &AlgeonNativeExecutionConfig) -> Arc<dyn ExecutionPlan> {
        let params = CugraphAlgorithmParams::TriangleCountAll(CugraphTriangleCountAllParams);
        let spec = CugraphAlgorithmSpec {
            algorithm: CugraphAlgorithm::TriangleCountAll,
            has_weight: true,
            has_edge_id: false,
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

    #[cfg(feature = "cugraph")]
    #[test]
    fn planner_ticket_aggregates_scopes_from_composed_native_fragments() {
        let (_backend, config) = backend_admission_config(64 * 1024 * 1024);
        let left = AlgeonNativeGpuProducerExec::try_new(
            "left",
            local_parquet_native_plan("/tmp/algeon-datafusion-planner-left-not-read.parquet")
                .clone_with_gpu_resident_sink(),
        )
        .unwrap();
        let right = AlgeonNativeGpuProducerExec::try_new(
            "right",
            local_parquet_native_plan("/tmp/algeon-datafusion-planner-right-not-read.parquet")
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

        let expected =
            composed_admission_capability_evidence(&consumer_plan, &[Arc::new(left.clone()), Arc::new(right.clone())])
                .expect("composed evidence binds both producer occurrences");
        let composed =
            AlgeonNativeComposedExec::try_new(consumer_plan, vec![Arc::new(left), Arc::new(right)], config.clone())
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
        let rule = AlgeonNativePhysicalOptimizerRule::new(optimizer_config);
        let selected_by_count_probe = native_exec_leaf();

        let final_plan = rule.optimize(selected_by_count_probe, &ConfigOptions::default()).unwrap();

        assert!(
            plan_is_query_admission_wrapped(final_plan.as_ref()),
            "the final pass must bind admission even when relational selection already happened"
        );
    }

    fn union_over_empty_with_unsupported_field(name: &str) -> Arc<dyn ExecutionPlan> {
        let schema: SchemaRef = Arc::new(Schema::new(vec![Field::new(
            name,
            DataType::Struct(Fields::from(vec![Arc::new(Field::new("x", DataType::Int64, true))])),
            true,
        )]));
        UnionExec::try_new(vec![Arc::new(EmptyExec::new(Arc::clone(&schema))), Arc::new(EmptyExec::new(schema))])
            .expect("union over empty inputs")
    }

    fn unsupported_column_name(diagnostics: &NativePlanningDiagnostics) -> &str {
        diagnostics
            .report
            .outcomes()
            .iter()
            .enumerate()
            .find_map(|(index, outcome)| match outcome {
                CandidateOutcome::NotSupported { .. } => diagnostics.report.unsupported_column(index),
                CandidateOutcome::Selected { .. }
                | CandidateOutcome::Deferred { .. }
                | CandidateOutcome::NotSelectedByCost { .. } => None,
            })
            .expect("unsupported column detail")
            .name
            .as_str()
    }

    /// Plans sharing one rule instance, two under their own invocations and
    /// one without any, each get a NativeRequired error and planning record
    /// that name only their own plan's column.
    #[test]
    fn native_required_errors_and_records_name_only_their_own_plan() {
        let rule = AlgeonNativePhysicalOptimizerRule::new(
            AlgeonNativeOptimizerConfig::default()
                .with_datafusion_cpu_execution_policy(DataFusionCpuExecutionPolicy::Reject),
        );
        let rejected_column = |plan, options: &ConfigOptions| {
            let failure = rule.optimize(plan, options).expect_err("NativeRequired rejects the CPU plan");
            failure
                .find_algeon_error()
                .expect("typed adapter error")
                .fact_value("unsupported_column")
                .map(str::to_string)
        };
        let invocation_a = Invocation::detached();
        let mut options_a = ConfigOptions::default();
        options_a.extensions.insert(invocation_a.clone());
        let invocation_b = Invocation::detached();
        let mut options_b = ConfigOptions::default();
        options_b.extensions.insert(invocation_b.clone());

        assert_eq!(
            rejected_column(union_over_empty_with_unsupported_field("a_col"), &options_a).as_deref(),
            Some("a_col")
        );
        assert_eq!(
            rejected_column(union_over_empty_with_unsupported_field("b_col"), &options_b).as_deref(),
            Some("b_col")
        );
        assert_eq!(
            rejected_column(union_over_empty_with_unsupported_field("plain_col"), &ConfigOptions::default()).as_deref(),
            Some("plain_col")
        );

        assert_eq!(
            unsupported_column_name(&invocation_a.planning_diagnostics().expect("invocation A record")),
            "a_col"
        );
        assert_eq!(
            unsupported_column_name(&invocation_b.planning_diagnostics().expect("invocation B record")),
            "b_col"
        );
    }

    /// The count-only probe and the final rewrite of one planning pass share
    /// the invocation's footers, so a query that both stages lower parses each
    /// file's footer once.
    #[tokio::test]
    async fn one_planning_pass_parses_each_footer_once_across_optimizer_stages() {
        use arrow::{array::Int64Array, record_batch::RecordBatch};
        use datafusion::{
            execution::SessionStateBuilder,
            prelude::{ParquetReadOptions, SessionContext},
        };
        use parquet::arrow::ArrowWriter;

        use crate::{native::footer_parses_on_this_thread, session::AlgeonSessionStateBuilderExt};

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("values.parquet");
        let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
        let batch =
            RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int64Array::from_iter_values(0..8))]).unwrap();
        let mut writer = ArrowWriter::try_new(std::fs::File::create(&path).unwrap(), schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
        let state = SessionStateBuilder::new_with_default_features()
            .with_algeon_native(AlgeonNativeOptimizerConfig::default())
            .build();
        let ctx = SessionContext::new_with_state(state);
        ctx.register_parquet("vals", path.to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();
        // The probe lowers the source before it declines a non-count aggregate,
        // and the final rewrite lowers it again.
        let logical = ctx.state().create_logical_plan("SELECT SUM(value) FROM vals").await.unwrap();

        let parses_before = footer_parses_on_this_thread();
        let (plan, _) = plan_with_native_diagnostics(&ctx.state(), &logical).await.unwrap();

        assert!(plan_contains_gpu_execution(plan.as_ref()), "the final rewrite lowers the aggregate");
        assert_eq!(footer_parses_on_this_thread() - parses_before, 1);
    }
}
