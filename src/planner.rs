use std::{
    fmt,
    sync::{Arc, Mutex},
};

use datafusion::{
    common::{Result as DataFusionResult, config::ConfigOptions},
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
    report::{GpuCandidate, PlanningReport},
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
            return Ok(plan);
        };

        let lowering_input = match self.phase {
            NativeOptimizerPhase::CountOnlyProbe => peel_output_requirement_exec(Arc::clone(&plan)),
            NativeOptimizerPhase::Final => Arc::clone(&plan),
        };
        let lowering = match self.phase {
            NativeOptimizerPhase::CountOnlyProbe => {
                crate::native::lower_count_only_physical_plan_to_native(lowering_input)
            },
            NativeOptimizerPhase::Final => crate::native::lower_physical_plan_to_native(Arc::clone(&plan)),
        };
        *self.state.last_report.lock().expect("native planning report mutex poisoned") = Some(lowering.report.clone());
        *self.state.last_source_diagnostics.lock().expect("native source diagnostics mutex poisoned") =
            lowering.source_diagnostics.clone();

        if lowering.plans.len() != 1 || lowering.rewritten_count() != 1 {
            return Ok(plan);
        }

        NativeCudfExec::try_new(lowering.plans[0].clone(), native_execution_config.clone())
            .map(|plan| Arc::new(plan) as Arc<dyn ExecutionPlan>)
            .map_err(Into::into)
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
        report.push_rejected(candidate.public, reason);
        if config.fallback_policy == GpuFallbackPolicy::GpuOnly {
            return Err(candidate_rejected_error(&candidate.public, reason, config).into());
        }
        return Ok(plan);
    }

    report.push_rewritten(candidate.public);
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
    .with_fact("candidate_shape", candidate.shape)
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
