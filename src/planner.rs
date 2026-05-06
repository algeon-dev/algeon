use std::{
    collections::BTreeMap,
    fmt,
    sync::{Arc, Mutex},
};

use datafusion::{
    common::{DataFusionError, Result as DataFusionResult, config::ConfigOptions},
    datasource::{physical_plan::ParquetSource, source::DataSourceExec},
    physical_optimizer::PhysicalOptimizerRule,
    physical_plan::{
        ExecutionPlan, aggregates::AggregateExec, filter::FilterExec, limit::LocalLimitExec,
        projection::ProjectionExec, sorts::sort::SortExec,
    },
};

use crate::{
    capability::{aggregate_rejection, expr_rejection, projection_rejection, sort_rejection},
    error::{Error, ErrorCode},
    exec::CudfExecutionConfig,
    native_exec::{NativeCudfExec, NativeCudfExecutionConfig},
    policy::{FallbackReason, GpuFallbackPolicy},
    report::{CandidateOutcome, GpuCandidate, PlanningReport},
    session::{
        CUDF_NATIVE_COUNT_ONLY_PHYSICAL_OPTIMIZER_RULE_NAME, CUDF_NATIVE_PHYSICAL_OPTIMIZER_RULE_NAME,
        CUDF_PHYSICAL_OPTIMIZER_RULE_NAME,
    },
};

#[derive(Debug, Clone)]
pub struct CudfOptimizerConfig {
    fallback_policy: GpuFallbackPolicy,
    min_operators_per_island: usize,
    execution_config: CudfExecutionConfig,
    native_execution_config: Option<NativeCudfExecutionConfig>,
}

impl Default for CudfOptimizerConfig {
    fn default() -> Self {
        Self {
            fallback_policy: GpuFallbackPolicy::default(),
            min_operators_per_island: 2,
            execution_config: CudfExecutionConfig::default(),
            native_execution_config: None,
        }
    }
}

impl CudfOptimizerConfig {
    pub fn new(fallback_policy: GpuFallbackPolicy) -> Self {
        Self { fallback_policy, ..Self::default() }
    }

    pub fn fallback_policy(&self) -> GpuFallbackPolicy {
        self.fallback_policy
    }

    pub fn min_operators_per_island(&self) -> usize {
        self.min_operators_per_island
    }

    pub fn execution_config(&self) -> &CudfExecutionConfig {
        &self.execution_config
    }

    pub fn native_execution_config(&self) -> Option<&NativeCudfExecutionConfig> {
        self.native_execution_config.as_ref()
    }

    pub fn native_execution_enabled(&self) -> bool {
        self.native_execution_config.is_some()
    }

    pub fn with_min_operators_per_island(mut self, min_operators_per_island: usize) -> Self {
        self.min_operators_per_island = min_operators_per_island.max(1);
        self
    }

    pub fn with_execution_config(mut self, execution_config: CudfExecutionConfig) -> Self {
        self.execution_config = execution_config;
        self
    }

    pub fn with_native_execution_config(mut self, native_execution_config: NativeCudfExecutionConfig) -> Self {
        self.native_execution_config = Some(native_execution_config);
        self
    }

    /// Use one caller-provided cudf runtime for GPU islands produced by this optimizer.
    pub fn with_runtime(mut self, runtime: cudf::Runtime) -> Self {
        self.execution_config = self.execution_config.clone().with_runtime(runtime);
        self
    }
}

#[derive(Debug, Clone)]
pub struct OptimizationResult {
    pub plan: Arc<dyn ExecutionPlan>,
    pub report: PlanningReport,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativePlanningDiagnostics {
    pub input_plan_root: String,
    pub input_plan_shape: String,
    pub report: PlanningReport,
    pub source_diagnostics: Vec<cudf_query_engine::SourceDiagnostics>,
    pub bounded_join_reservation_facts: BTreeMap<String, String>,
}

#[derive(Debug)]
pub struct CudfNativePhysicalOptimizerRule {
    config: CudfOptimizerConfig,
    phase: NativeOptimizerPhase,
    state: Arc<CudfNativePhysicalOptimizerRuleState>,
}

#[derive(Debug)]
struct CudfNativePhysicalOptimizerRuleState {
    last_input_plan_root: Mutex<Option<String>>,
    last_input_plan_shape: Mutex<Option<String>>,
    last_report: Mutex<Option<PlanningReport>>,
    last_source_diagnostics: Mutex<Vec<cudf_query_engine::SourceDiagnostics>>,
    last_bounded_join_reservation_facts: Mutex<BTreeMap<String, String>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NativeOptimizerPhase {
    CountOnlyProbe,
    Final,
}

impl CudfNativePhysicalOptimizerRule {
    pub fn new(config: CudfOptimizerConfig) -> Self {
        Self {
            config,
            phase: NativeOptimizerPhase::Final,
            state: Arc::new(CudfNativePhysicalOptimizerRuleState {
                last_input_plan_root: Mutex::new(None),
                last_input_plan_shape: Mutex::new(None),
                last_report: Mutex::new(None),
                last_source_diagnostics: Mutex::new(Vec::new()),
                last_bounded_join_reservation_facts: Mutex::new(BTreeMap::new()),
            }),
        }
    }

    pub fn config(&self) -> &CudfOptimizerConfig {
        &self.config
    }

    pub(crate) fn count_only_probe_rule(&self) -> Self {
        Self {
            config: self.config.clone(),
            phase: NativeOptimizerPhase::CountOnlyProbe,
            state: Arc::clone(&self.state),
        }
    }

    pub fn last_report(&self) -> Option<PlanningReport> {
        self.state.last_report.lock().expect("native planning report mutex poisoned").clone()
    }

    pub fn last_source_diagnostics(&self) -> Vec<cudf_query_engine::SourceDiagnostics> {
        self.state.last_source_diagnostics.lock().expect("native source diagnostics mutex poisoned").clone()
    }

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
            report,
            source_diagnostics: self.last_source_diagnostics(),
            bounded_join_reservation_facts: self
                .state
                .last_bounded_join_reservation_facts
                .lock()
                .expect("native bounded join reservation facts mutex poisoned")
                .clone(),
        })
    }
}

impl PhysicalOptimizerRule for CudfNativePhysicalOptimizerRule {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        if plan.as_any().downcast_ref::<NativeCudfExec>().is_some() {
            return Ok(plan);
        }
        if let Some(native_plan) = peel_cooperative_native_exec(&plan) {
            return Ok(native_plan);
        }

        *self.state.last_input_plan_root.lock().expect("native input plan root mutex poisoned") =
            Some(plan.name().to_string());
        *self.state.last_input_plan_shape.lock().expect("native input plan shape mutex poisoned") =
            Some(physical_plan_shape(plan.as_ref()));

        let Some(native_execution_config) = self.config.native_execution_config() else {
            *self.state.last_report.lock().expect("native planning report mutex poisoned") =
                Some(PlanningReport::new());
            *self.state.last_source_diagnostics.lock().expect("native source diagnostics mutex poisoned") = Vec::new();
            *self
                .state
                .last_bounded_join_reservation_facts
                .lock()
                .expect("native bounded join reservation facts mutex poisoned") = BTreeMap::new();
            return Ok(plan);
        };

        let optimized = match self.phase {
            NativeOptimizerPhase::CountOnlyProbe => {
                optimize_count_only_native_plan(Arc::clone(&plan), native_execution_config)?
            },
            NativeOptimizerPhase::Final => optimize_final_native_plan(Arc::clone(&plan), native_execution_config)?,
        };
        *self.state.last_report.lock().expect("native planning report mutex poisoned") = Some(optimized.report.clone());
        *self.state.last_source_diagnostics.lock().expect("native source diagnostics mutex poisoned") =
            optimized.source_diagnostics.clone();
        *self
            .state
            .last_bounded_join_reservation_facts
            .lock()
            .expect("native bounded join reservation facts mutex poisoned") =
            optimized.bounded_join_reservation_facts.clone();
        Ok(optimized.plan)
    }

    fn name(&self) -> &str {
        match self.phase {
            NativeOptimizerPhase::CountOnlyProbe => CUDF_NATIVE_COUNT_ONLY_PHYSICAL_OPTIMIZER_RULE_NAME,
            NativeOptimizerPhase::Final => CUDF_NATIVE_PHYSICAL_OPTIMIZER_RULE_NAME,
        }
    }

    fn schema_check(&self) -> bool {
        true
    }
}

#[derive(Debug, Clone)]
struct NativeOptimizationAttempt {
    plan: Arc<dyn ExecutionPlan>,
    report: PlanningReport,
    source_diagnostics: Vec<cudf_query_engine::SourceDiagnostics>,
    bounded_join_reservation_facts: BTreeMap<String, String>,
    rewritten: bool,
}

fn optimize_count_only_native_plan(
    plan: Arc<dyn ExecutionPlan>,
    native_execution_config: &NativeCudfExecutionConfig,
) -> DataFusionResult<NativeOptimizationAttempt> {
    let lowering_input = peel_output_requirement_exec(Arc::clone(&plan));
    let lowering = crate::native::lower_count_only_physical_plan_to_native(lowering_input);
    replace_lowered_root_or_keep_plan(plan, lowering, native_execution_config)
}

fn optimize_final_native_plan(
    plan: Arc<dyn ExecutionPlan>,
    native_execution_config: &NativeCudfExecutionConfig,
) -> DataFusionResult<NativeOptimizationAttempt> {
    let lowering = crate::native::lower_physical_plan_to_native(Arc::clone(&plan));
    if lowering.plans.len() == 1 && lowering.rewritten_count() == 1 {
        return replace_lowered_root_or_keep_plan(plan, lowering, native_execution_config);
    }

    let mut report = lowering.report.clone();
    let mut source_diagnostics = lowering.source_diagnostics.clone();
    let mut bounded_join_reservation_facts = BTreeMap::new();
    let Some(rewritten_plan) = optimize_native_children_for_host_arrow_boundary(
        Arc::clone(&plan),
        native_execution_config,
        &mut report,
        &mut source_diagnostics,
        &mut bounded_join_reservation_facts,
    )?
    else {
        return Ok(NativeOptimizationAttempt {
            plan,
            report,
            source_diagnostics,
            bounded_join_reservation_facts,
            rewritten: false,
        });
    };

    Ok(NativeOptimizationAttempt {
        plan: rewritten_plan,
        report,
        source_diagnostics,
        bounded_join_reservation_facts,
        rewritten: true,
    })
}

fn replace_lowered_root_or_keep_plan(
    plan: Arc<dyn ExecutionPlan>,
    lowering: crate::native::NativeLoweringResult,
    native_execution_config: &NativeCudfExecutionConfig,
) -> DataFusionResult<NativeOptimizationAttempt> {
    if lowering.plans.len() != 1 || lowering.rewritten_count() != 1 {
        return Ok(NativeOptimizationAttempt {
            plan,
            report: lowering.report,
            source_diagnostics: lowering.source_diagnostics,
            bounded_join_reservation_facts: BTreeMap::new(),
            rewritten: false,
        });
    }
    if let Some(bounded_join_reservation_facts) =
        native_execution_boundary_rejection_facts(&lowering.plans[0], native_execution_config)
    {
        return Ok(NativeOptimizationAttempt {
            plan,
            report: native_execution_boundary_report(&lowering.report),
            source_diagnostics: lowering.source_diagnostics,
            bounded_join_reservation_facts,
            rewritten: false,
        });
    }

    let plan = NativeCudfExec::try_new(lowering.plans[0].clone(), native_execution_config.clone())
        .map(|plan| Arc::new(plan) as Arc<dyn ExecutionPlan>)
        .map_err(DataFusionError::from)?;
    Ok(NativeOptimizationAttempt {
        plan,
        report: lowering.report,
        source_diagnostics: lowering.source_diagnostics,
        bounded_join_reservation_facts: BTreeMap::new(),
        rewritten: true,
    })
}

fn optimize_native_children_for_host_arrow_boundary(
    plan: Arc<dyn ExecutionPlan>,
    native_execution_config: &NativeCudfExecutionConfig,
    report: &mut PlanningReport,
    source_diagnostics: &mut Vec<cudf_query_engine::SourceDiagnostics>,
    bounded_join_reservation_facts: &mut BTreeMap<String, String>,
) -> DataFusionResult<Option<Arc<dyn ExecutionPlan>>> {
    if !host_arrow_boundary_allows_child_replacement(plan.as_ref()) {
        return Ok(None);
    }

    let children = plan.children();
    if children.is_empty() {
        return Ok(None);
    }

    let mut rewritten = false;
    let mut child_reports = Vec::new();
    let mut child_source_diagnostics = Vec::new();
    let mut new_children = Vec::with_capacity(children.len());
    for child in children {
        let attempt = optimize_final_native_plan(Arc::clone(child), native_execution_config)?;
        rewritten |= attempt.rewritten;
        child_reports.push(attempt.report);
        child_source_diagnostics.extend(attempt.source_diagnostics);
        bounded_join_reservation_facts.extend(attempt.bounded_join_reservation_facts);
        new_children.push(attempt.plan);
    }

    if !rewritten {
        return Ok(None);
    }

    for child_report in child_reports {
        append_planning_report(report, &child_report);
    }
    source_diagnostics.extend(child_source_diagnostics);
    plan.with_new_children(new_children).map(Some)
}

fn append_planning_report(report: &mut PlanningReport, extra: &PlanningReport) {
    *report = PlanningReport::from_outcomes(report.outcomes().iter().cloned().chain(extra.outcomes().iter().cloned()));
}

fn host_arrow_boundary_allows_child_replacement(plan: &dyn ExecutionPlan) -> bool {
    plan.as_any().downcast_ref::<FilterExec>().is_some()
        || plan.as_any().downcast_ref::<ProjectionExec>().is_some()
        || plan.as_any().downcast_ref::<SortExec>().is_some()
        || plan.as_any().downcast_ref::<LocalLimitExec>().is_some()
        || plan.as_any().downcast_ref::<AggregateExec>().is_some()
}

fn native_execution_boundary_rejection_facts(
    plan: &cudf_query_engine::QueryPlan,
    execution_config: &NativeCudfExecutionConfig,
) -> Option<BTreeMap<String, String>> {
    if execution_config.memory_control_policy().is_bounded()
        && plan.contains_join()
        && !plan.has_bounded_executable_join_pipeline()
    {
        let mut facts: BTreeMap<String, String> = cudf_query_engine::BoundedJoinReservationContract
            .rejection_facts(execution_config.boundary_policy())
            .into_iter()
            .map(|(key, value)| (key.to_string(), value))
            .collect();
        facts.insert("bounded_join_execution_status".to_string(), "unsupported_plan_shape".to_string());
        facts.insert("bounded_join_unsupported_shape".to_string(), "requires_root_reusable_hash_join".to_string());
        Some(facts)
    } else {
        None
    }
}

fn native_execution_boundary_report(report: &PlanningReport) -> PlanningReport {
    PlanningReport::from_outcomes(report.outcomes().iter().map(|outcome| match outcome {
        CandidateOutcome::Rewritten { candidate } => CandidateOutcome::Rejected {
            candidate: candidate.clone(),
            reason: FallbackReason::NativeBoundedJoinUnsupported,
        },
        CandidateOutcome::Rejected { candidate, reason } => {
            CandidateOutcome::Rejected { candidate: candidate.clone(), reason: *reason }
        },
    }))
}

fn peel_output_requirement_exec(plan: Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
    if plan.name() == "OutputRequirementExec"
        && let [child] = plan.children().as_slice()
    {
        return Arc::clone(child);
    }
    plan
}

fn peel_cooperative_native_exec(plan: &Arc<dyn ExecutionPlan>) -> Option<Arc<dyn ExecutionPlan>> {
    if plan.name() != "CooperativeExec" {
        return None;
    }
    let children = plan.children();
    let [child] = children.as_slice() else {
        return None;
    };
    child.as_any().downcast_ref::<NativeCudfExec>().is_some().then(|| Arc::clone(child))
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

#[derive(Debug)]
pub struct CudfPhysicalOptimizerRule {
    config: CudfOptimizerConfig,
    last_report: Mutex<Option<PlanningReport>>,
}

impl CudfPhysicalOptimizerRule {
    pub fn new(config: CudfOptimizerConfig) -> Self {
        Self { config, last_report: Mutex::new(None) }
    }

    pub fn config(&self) -> &CudfOptimizerConfig {
        &self.config
    }

    pub fn last_report(&self) -> Option<PlanningReport> {
        self.last_report.lock().expect("planning report mutex poisoned").clone()
    }
}

impl PhysicalOptimizerRule for CudfPhysicalOptimizerRule {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let OptimizationResult { plan, report } = optimize_plan(plan, &self.config)?;
        *self.last_report.lock().expect("planning report mutex poisoned") = Some(report);
        Ok(plan)
    }

    fn name(&self) -> &str {
        CUDF_PHYSICAL_OPTIMIZER_RULE_NAME
    }

    fn schema_check(&self) -> bool {
        true
    }
}

pub fn optimize_plan(
    plan: Arc<dyn ExecutionPlan>,
    config: &CudfOptimizerConfig,
) -> DataFusionResult<OptimizationResult> {
    let mut report = PlanningReport::new();
    let plan = optimize_node(plan, config, &mut report)?;
    Ok(OptimizationResult { plan, report })
}

fn optimize_node(
    plan: Arc<dyn ExecutionPlan>,
    config: &CudfOptimizerConfig,
    report: &mut PlanningReport,
) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
    if let Some(candidate) = inspect_composite_candidate(plan.as_ref()) {
        return handle_candidate(plan, candidate, config, report);
    }

    let children = plan.children();
    let plan = if children.is_empty() {
        plan
    } else {
        let new_children = children
            .into_iter()
            .map(|child| optimize_node(Arc::clone(child), config, report))
            .collect::<DataFusionResult<Vec<_>>>()?;
        plan.with_new_children(new_children)?
    };

    let Some(candidate) = inspect_candidate(plan.as_ref()) else {
        return Ok(plan);
    };

    handle_candidate(plan, candidate, config, report)
}

fn handle_candidate(
    plan: Arc<dyn ExecutionPlan>,
    candidate: InspectedCandidate,
    config: &CudfOptimizerConfig,
    report: &mut PlanningReport,
) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
    let rejection = rejection_reason(&candidate, config);
    if let Some(reason) = rejection {
        report.push_rejected(candidate.public.clone(), reason);
        if config.fallback_policy == GpuFallbackPolicy::GpuOnly {
            return Err(candidate_rejected_error(&candidate.public, reason, config).into());
        }
        return Ok(plan);
    }

    report.push_rewritten(candidate.public.clone());
    crate::exec::CudfGpuIslandExec::try_new_from_plan_with_execution_config(plan, config.execution_config().clone())
        .map(|plan| Arc::new(plan) as Arc<dyn ExecutionPlan>)
        .map_err(Into::into)
}

fn rejection_reason(candidate: &InspectedCandidate, config: &CudfOptimizerConfig) -> Option<FallbackReason> {
    if config.fallback_policy == GpuFallbackPolicy::CpuOnly {
        return Some(FallbackReason::ExplicitCpuPolicy);
    }
    if let Some(reason) = candidate.rejection {
        return Some(reason);
    }
    if candidate.public.operator_count < config.min_operators_per_island {
        return Some(FallbackReason::InsufficientIslandSize);
    }
    None
}

fn candidate_rejected_error(candidate: &GpuCandidate, reason: FallbackReason, config: &CudfOptimizerConfig) -> Error {
    Error::new(
        ErrorCode::GpuCandidateRejected,
        "datafusion_cudf.optimize_plan",
        "GPU candidate rejected by adapter policy",
    )
    .with_fact("candidate_shape", candidate.shape.as_str())
    .with_fact("fallback_reason", reason.code())
    .with_fact("fallback_policy", config.fallback_policy())
    .with_fact("operator_count", candidate.operator_count)
    .with_fact("min_operators_per_island", config.min_operators_per_island())
}

#[derive(Debug, Clone)]
struct InspectedCandidate {
    public: GpuCandidate,
    rejection: Option<FallbackReason>,
}

fn inspect_composite_candidate(plan: &dyn ExecutionPlan) -> Option<InspectedCandidate> {
    if let Some(limit) = plan.as_any().downcast_ref::<LocalLimitExec>() {
        if let Some(sort) = limit.input().as_any().downcast_ref::<SortExec>() {
            if let Some(aggregate) = sort.input().as_any().downcast_ref::<AggregateExec>() {
                let projection = aggregate.input().as_any().downcast_ref::<ProjectionExec>()?;
                let filter = projection.input().as_any().downcast_ref::<FilterExec>()?;
                let rejection = first_rejection([
                    aggregate_rejection(aggregate),
                    sort_rejection(sort),
                    projection_rejection(projection),
                    expr_rejection(filter.predicate()),
                ]);
                return Some(InspectedCandidate {
                    public: GpuCandidate::with_operator_count(
                        "LocalLimitExec(SortExec(AggregateExec(ProjectionExec(FilterExec(child)))))",
                        sort.schema().fields().len(),
                        5,
                    ),
                    rejection,
                });
            }

            let projection = sort.input().as_any().downcast_ref::<ProjectionExec>()?;
            let filter = projection.input().as_any().downcast_ref::<FilterExec>()?;
            let rejection = first_rejection([
                projection_rejection(projection),
                sort_rejection(sort),
                expr_rejection(filter.predicate()),
            ]);
            return Some(InspectedCandidate {
                public: GpuCandidate::with_operator_count(
                    "LocalLimitExec(SortExec(ProjectionExec(FilterExec(child))))",
                    sort.schema().fields().len(),
                    4,
                ),
                rejection,
            });
        }

        let projection = limit.input().as_any().downcast_ref::<ProjectionExec>()?;
        let filter = projection.input().as_any().downcast_ref::<FilterExec>()?;
        let rejection = first_rejection([projection_rejection(projection), expr_rejection(filter.predicate())]);
        return Some(InspectedCandidate {
            public: GpuCandidate::with_operator_count(
                "LocalLimitExec(ProjectionExec(FilterExec(child)))",
                projection.schema().fields().len(),
                3,
            ),
            rejection,
        });
    }

    if let Some(sort) = plan.as_any().downcast_ref::<SortExec>() {
        if let Some(aggregate) = sort.input().as_any().downcast_ref::<AggregateExec>() {
            let projection = aggregate.input().as_any().downcast_ref::<ProjectionExec>()?;
            let filter = projection.input().as_any().downcast_ref::<FilterExec>()?;
            let rejection = first_rejection([
                aggregate_rejection(aggregate),
                sort_rejection(sort),
                projection_rejection(projection),
                expr_rejection(filter.predicate()),
            ]);
            return Some(InspectedCandidate {
                public: GpuCandidate::with_operator_count(
                    "SortExec(AggregateExec(ProjectionExec(FilterExec(child))))",
                    sort.schema().fields().len(),
                    4,
                ),
                rejection,
            });
        }

        let projection = sort.input().as_any().downcast_ref::<ProjectionExec>()?;
        let filter = projection.input().as_any().downcast_ref::<FilterExec>()?;
        let rejection = first_rejection([
            projection_rejection(projection),
            sort_rejection(sort),
            expr_rejection(filter.predicate()),
        ]);
        return Some(InspectedCandidate {
            public: GpuCandidate::with_operator_count(
                "SortExec(ProjectionExec(FilterExec(child)))",
                sort.schema().fields().len(),
                3,
            ),
            rejection,
        });
    }

    if let Some(aggregate) = plan.as_any().downcast_ref::<AggregateExec>() {
        let projection = aggregate.input().as_any().downcast_ref::<ProjectionExec>()?;
        let filter = projection.input().as_any().downcast_ref::<FilterExec>()?;
        let rejection = first_rejection([
            aggregate_rejection(aggregate),
            projection_rejection(projection),
            expr_rejection(filter.predicate()),
        ]);
        return Some(InspectedCandidate {
            public: GpuCandidate::with_operator_count(
                "AggregateExec(ProjectionExec(FilterExec(child)))",
                aggregate.schema().fields().len(),
                3,
            ),
            rejection,
        });
    }

    let projection = plan.as_any().downcast_ref::<ProjectionExec>()?;
    let filter = projection.input().as_any().downcast_ref::<FilterExec>()?;
    let rejection = first_rejection([projection_rejection(projection), expr_rejection(filter.predicate())]);
    Some(InspectedCandidate {
        public: GpuCandidate::with_operator_count(
            "ProjectionExec(FilterExec(child))",
            projection.schema().fields().len(),
            2,
        ),
        rejection,
    })
}

fn inspect_candidate(plan: &dyn ExecutionPlan) -> Option<InspectedCandidate> {
    if let Some(data_source) = plan.as_any().downcast_ref::<DataSourceExec>()
        && data_source.downcast_to_file_source::<ParquetSource>().is_some()
    {
        return Some(InspectedCandidate {
            public: GpuCandidate::with_operator_count("LocalParquetScan", data_source.schema().fields().len(), 1),
            rejection: None,
        });
    }

    if let Some(projection) = plan.as_any().downcast_ref::<ProjectionExec>() {
        return Some(InspectedCandidate {
            public: GpuCandidate::with_operator_count("ProjectionExec(child)", projection.schema().fields().len(), 1),
            rejection: projection_rejection(projection),
        });
    }

    if let Some(filter) = plan.as_any().downcast_ref::<FilterExec>() {
        return Some(InspectedCandidate {
            public: GpuCandidate::with_operator_count("FilterExec(child)", filter.schema().fields().len(), 1),
            rejection: expr_rejection(filter.predicate()),
        });
    }

    None
}

fn first_rejection(rejections: impl IntoIterator<Item = Option<FallbackReason>>) -> Option<FallbackReason> {
    rejections.into_iter().flatten().next()
}

impl fmt::Display for GpuFallbackPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::GpuOnly => f.write_str("GpuOnly"),
            Self::GpuPreferred => f.write_str("GpuPreferred"),
            Self::CpuOnly => f.write_str("CpuOnly"),
        }
    }
}
