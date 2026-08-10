use std::sync::{Arc, Mutex};

use datafusion::{
    common::{DataFusionError, Result as DataFusionResult, config::ConfigOptions},
    physical_optimizer::PhysicalOptimizerRule,
    physical_plan::ExecutionPlan,
};
use nexus_query_engine::{admission::QueryPlanCapabilityProof, metrics::QueryId};

use super::{FinalPlanDisposition, FinalPlanRequirement, NexusNativeOptimizerConfig, classify_final_plan};
use crate::{
    attempt::NexusAttemptExtension,
    error::{Error, ErrorCode},
    native::lakehouse_pruning,
    native_exec::{
        NexusNativeComposedExec, NexusNativeComposedGpuProducerExec, NexusNativeExec, NexusNativeExecutionConfig,
        NexusNativeGpuProducerExec,
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
    NativeCapabilityRejectedCandidateDiagnostics,
};
use mixed::is_certified_mixed_execution_plan;

/// Internal optimizer-rule identity for completed-plan requirements.
pub const NEXUS_FINAL_PLAN_REQUIREMENT_RULE_NAME: &str = "datafusion_nexus_final_plan_requirement";

/// Enforces configured requirements against the exact physical plan produced by
/// the preceding native rewrite and leaf-guard rules.
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
        if self.config.final_plan_requirements().is_empty() {
            return Ok(plan);
        }
        let disposition = classify_final_plan(plan.as_ref());
        // A feature-disabled cuGraph plan is a zero-child terminal capability
        // rejection. It executes neither a CPU fallback nor a GPU operator,
        // so it satisfies the no-CPU-execution contract without being
        // misclassified as a native workload.
        let feature_disabled_terminal = crate::cugraph_sql::is_feature_disabled_execution_plan(plan.as_ref());
        for requirement in self.config.final_plan_requirements() {
            if *requirement == FinalPlanRequirement::NoDataFusionCpu
                && disposition != FinalPlanDisposition::Native
                && !feature_disabled_terminal
            {
                return Err(final_plan_requirement_error(
                    plan.as_ref(),
                    disposition,
                    *requirement,
                    self.planning_report().as_ref(),
                )
                .into());
            }
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
    requirement: FinalPlanRequirement,
    report: Option<&PlanningReport>,
) -> Error {
    let mut error = Error::new(
        ErrorCode::FinalPlanRequirementUnsatisfied,
        "datafusion_nexus.planner.final_plan_requirement",
        "exact final physical plan does not satisfy the configured GPU execution requirement",
    )
    .with_fact("requirement", requirement.as_str())
    .with_fact("observed_final_plan_disposition", disposition.as_str())
    .with_fact("terminal_plan_disposition", FinalPlanDisposition::Rejected.as_str())
    .with_fact("physical_plan_root", plan.name());
    if let Some((candidate, capability_reason)) = report.and_then(first_native_not_supported) {
        error = error
            .with_fact("candidate_shape", candidate.shape.as_str())
            .with_fact("capability_reason", capability_reason.code())
            .with_fact("operator_count", candidate.operator_count);
    }
    error
}

fn first_native_not_supported(
    report: &PlanningReport,
) -> Option<(&crate::report::GpuCandidate, crate::capability::CapabilityReason)> {
    report.outcomes().iter().find_map(|outcome| match outcome {
        CandidateOutcome::NotSupported { candidate, capability_reason } => Some((candidate, *capability_reason)),
        CandidateOutcome::Selected { .. } | CandidateOutcome::NotSelectedByCost { .. } => None,
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
/// Candidate misses leave the original DataFusion plan intact. Optional
/// completed-plan requirements run in a later rule after native planning has
/// completed.
#[derive(Debug)]
pub struct NexusNativePhysicalOptimizerRule {
    config: NexusNativeOptimizerConfig,
    stage: NativeOptimizerStage,
    state: Arc<NexusNativePhysicalOptimizerRuleState>,
}

#[derive(Debug)]
struct NexusNativePhysicalOptimizerRuleState {
    last_input_plan_root: Mutex<Option<String>>,
    last_input_plan_shape: Mutex<Option<String>>,
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
                last_input_plan_root: Mutex::new(None),
                last_input_plan_shape: Mutex::new(None),
                last_input_join_diagnostics: Mutex::new(Vec::new()),
                last_report: Mutex::new(None),
                last_native_optimizer: Mutex::new(nexus_query_engine::optimizer::NativeOptimizerDiagnostics::default()),
                last_source_diagnostics: Mutex::new(Vec::new()),
                last_capability_contract: Mutex::new(NativeCapabilityContractDiagnostics::default()),
            }),
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
        }
    }

    /// Most recent planning report produced by this retained rule, if it has run.
    pub fn last_report(&self) -> Option<PlanningReport> {
        match self.state.last_report.lock() {
            Ok(report) => report.clone(),
            Err(poisoned) => {
                tracing::warn!("native planning report mutex was poisoned; returning the last captured report");
                poisoned.into_inner().clone()
            },
        }
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
        if plan_is_query_admission_wrapped(plan.as_ref()) {
            return Ok(plan);
        }
        if plan.downcast_ref::<NexusNativeExec>().is_some() || plan.downcast_ref::<NexusNativeComposedExec>().is_some()
        {
            return Ok(plan);
        }
        if let Some(native_plan) = peel_cooperative_native_exec(&plan) {
            return Ok(native_plan);
        }
        if is_certified_mixed_execution_plan(plan.as_ref()) {
            return Ok(plan);
        }

        *self.state.last_input_plan_root.lock().expect("native input plan root mutex poisoned") =
            Some(plan.name().to_string());
        *self.state.last_input_plan_shape.lock().expect("native input plan shape mutex poisoned") =
            Some(physical_plan_shape(plan.as_ref()));
        *self.state.last_input_join_diagnostics.lock().expect("native input join diagnostics mutex poisoned") =
            physical_join_diagnostics(plan.as_ref());

        let attempt_extension = config.extensions.get::<NexusAttemptExtension>().cloned();
        let effective_config = self.config.with_datafusion_config_options(config)?.with_query_id_if_absent(
            attempt_extension
                .as_ref()
                .and_then(NexusAttemptExtension::query_id)
                .map(|query_id| QueryId::new(query_id.as_str()))
                .unwrap_or_else(QueryId::mint),
        );
        let native_execution_config = effective_config
            .native_execution_config()
            .clone()
            .with_lakehouse_footer_pruning(lakehouse_pruning::enabled_from_env());
        let effective_config = effective_config.with_native_execution_config(native_execution_config);
        let native_execution_config = effective_config.native_execution_config();
        let native_optimizer_options = effective_config.native_optimizer_options();
        let native_plan_cost_options = effective_config.native_plan_cost_options();

        let optimized = match self.stage {
            NativeOptimizerStage::CountOnlyProbe => optimize_count_only_native_plan(
                Arc::clone(&plan),
                native_execution_config,
                native_optimizer_options,
                native_plan_cost_options,
            )?,
            NativeOptimizerStage::Final => optimize_final_native_plan(
                Arc::clone(&plan),
                native_execution_config,
                native_optimizer_options,
                native_plan_cost_options,
            )?,
        };
        let mut optimized = optimized;
        if optimized.rewritten
            && let Some(extension) = &attempt_extension
        {
            extension.native_admission_planned(optimized.plan.as_ref());
        }
        let query_admission_ticket = if optimized.rewritten {
            plan_query_admission_ticket_for_optimized_plan(
                optimized.plan.as_ref(),
                native_execution_config,
                attempt_extension.as_ref(),
            )?
        } else {
            None
        };
        if let (Some(extension), Some(ticket)) = (&attempt_extension, &query_admission_ticket) {
            extension.bind_ticket(Arc::clone(ticket));
        }
        apply_execution_admission_wrappers(&mut optimized, query_admission_ticket);
        self.store_optimization_attempt(&optimized)?;
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
    if optimized.rewritten
        && let Some(ticket) = query_admission_ticket
    {
        optimized.plan = wrap_plan_with_query_admission_ticket(Arc::clone(&optimized.plan), ticket);
    }
}

fn plan_query_admission_ticket_for_optimized_plan(
    plan: &dyn ExecutionPlan,
    native_execution_config: &NexusNativeExecutionConfig,
    attempt: Option<&NexusAttemptExtension>,
) -> DataFusionResult<Option<Arc<QueryAdmissionTicket>>> {
    // Frontend graph roots may resolve lakehouse footer metadata asynchronously
    // before admission.  A planner-time ticket would freeze the incomplete
    // cardinality evidence, so that root mints its sole immutable ticket only
    // after the metadata is bound to the exact execution plan.
    if plan_contains_deferred_graph_admission(plan) {
        return Ok(None);
    }
    let fragment_proofs = native_fragment_capability_proofs(plan)?;
    let ticket = native_execution_config
        .new_plan_query_admission_ticket_with_fragment_requirements(fragment_proofs)
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
            ticket.attempt_capability_proof_for_rebind(),
            extension.ticket_binding(),
            ticket.metrics_capture(),
        ),
        None => ticket,
    };
    Ok(Some(ticket))
}

fn plan_contains_deferred_graph_admission(plan: &dyn ExecutionPlan) -> bool {
    graph_execution_plan(plan).is_some()
        || plan.children().iter().any(|child| plan_contains_deferred_graph_admission(child.as_ref()))
}

fn native_fragment_capability_proofs(plan: &dyn ExecutionPlan) -> DataFusionResult<Vec<QueryPlanCapabilityProof>> {
    let mut proofs = Vec::new();
    collect_native_fragment_capability_proofs(plan, false, &mut proofs)?;
    Ok(proofs)
}

fn collect_native_fragment_capability_proofs(
    plan: &dyn ExecutionPlan,
    executes_as_cugraph_input: bool,
    proofs: &mut Vec<QueryPlanCapabilityProof>,
) -> DataFusionResult<()> {
    let graph_execution_proof =
        graph_execution_plan(plan).map(GraphExecutionPlan::graph_execution_capability_proof).transpose()?;
    let is_cugraph_execution = graph_execution_proof.is_some();
    for child in plan.children() {
        collect_native_fragment_capability_proofs(
            child.as_ref(),
            executes_as_cugraph_input || is_cugraph_execution,
            proofs,
        )?;
    }
    if let Some(proof) = graph_execution_proof {
        proofs.push(proof);
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
        proofs.push(native_plan_capability_proof(native_plan)?);
    }
    Ok(())
}

fn native_plan_capability_proof(
    native_plan: &nexus_query_engine::plan::QueryPlan,
) -> DataFusionResult<QueryPlanCapabilityProof> {
    QueryPlanCapabilityProof::for_query_plan_preflighted(native_plan).map_err(|error| {
        DataFusionError::from(Error::from_native(
            "datafusion_nexus.native_optimizer.admission_proof",
            "native query-plan preflight failed before admission proof issuance",
            error,
        ))
    })
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
        assert_eq!(FinalPlanDisposition::Rejected.as_str(), "rejected");
    }

    #[test]
    fn no_datafusion_cpu_requirement_rejects_the_exact_datafusion_final_plan() {
        let plan: Arc<dyn ExecutionPlan> = Arc::new(EmptyExec::new(Arc::new(Schema::empty())));
        let rule = NexusFinalPlanRequirementRule::new(
            NexusNativeOptimizerConfig::default().with_no_datafusion_cpu_requirement(),
        );

        let error = rule
            .optimize(plan, &ConfigOptions::default())
            .expect_err("the requirement must reject a CPU-only final plan");
        let error = error.find_nexus_error().expect("typed adapter error");

        assert_eq!(error.code(), ErrorCode::FinalPlanRequirementUnsatisfied);
        assert_eq!(error.status(), ErrorStatus::Permanent);
        assert_eq!(error.fact_value("requirement"), Some("no_datafusion_cpu"));
        assert_eq!(error.fact_value("observed_final_plan_disposition"), Some("datafusion"));
        assert_eq!(error.fact_value("terminal_plan_disposition"), Some("rejected"));
    }

    #[test]
    fn empty_final_plan_requirement_set_returns_the_original_plan_without_replanning() {
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
    fn planner_ticket_unions_capabilities_from_composed_native_fragments() {
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

        let left_proof = native_plan_capability_proof(left.native_plan()).expect("producer has capability proof");
        let right_proof = native_plan_capability_proof(right.native_plan()).expect("producer has capability proof");
        let consumer_proof = native_plan_capability_proof(&consumer_plan).expect("consumer has capability proof");
        let unaggregated_capability_count = left_proof.required_capabilities().len()
            + right_proof.required_capabilities().len()
            + consumer_proof.required_capabilities().len();
        assert!(!left_proof.required_capabilities().is_empty(), "producer must expose a capability requirement");
        let composed =
            NexusNativeComposedExec::try_new(consumer_plan, vec![Arc::new(left), Arc::new(right)], config.clone())
                .unwrap();
        let expected = QueryPlanCapabilityProof::combine(left_proof, [right_proof, consumer_proof]).unwrap();

        let ticket = plan_query_admission_ticket_for_optimized_plan(&composed, &config, None).unwrap().unwrap();

        assert_eq!(ticket.attempt_capability_proof(), Some(&expected));
        assert!(
            ticket.attempt_capability_proof().expect("planner ticket retains its proof").required_capabilities().len()
                < unaggregated_capability_count,
            "the repeated producer capability must be deduplicated"
        );
    }

    #[cfg(feature = "cugraph")]
    #[test]
    fn planner_defers_outer_ticket_for_registered_graph_descendant() {
        let (_backend, config) = backend_admission_config(64 * 1024 * 1024);
        let graph = deferred_cugraph_exec(&config);
        let plan = Arc::new(CooperativeExec::new(graph)) as Arc<dyn ExecutionPlan>;

        assert!(plan_contains_deferred_graph_admission(plan.as_ref()));
        assert!(
            plan_query_admission_ticket_for_optimized_plan(plan.as_ref(), &config, None)
                .expect("registered cuGraph discovery should not fail")
                .is_none(),
            "the planner must not wrap a graph descendant before footer metadata is hydrated",
        );
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
}
