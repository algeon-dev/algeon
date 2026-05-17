use arrow::datatypes::Schema;
use datafusion::{
    datasource::source::DataSourceExec,
    logical_expr::JoinType,
    physical_expr::{PhysicalExpr, ScalarFunctionExpr, expressions::BinaryExpr},
    physical_plan::{
        coalesce_partitions::CoalescePartitionsExec,
        joins::{CrossJoinExec, HashJoinExec, NestedLoopJoinExec},
        limit::GlobalLimitExec,
        repartition::RepartitionExec,
        union::UnionExec,
        windows::{BoundedWindowAggExec, WindowAggExec},
    },
};

use super::*;

mod adaptive;
mod strict_bounded;

pub use adaptive::{NativeAdaptivePipelineCandidateDiagnostics, NativeAdaptivePipelineDiagnostics};
use adaptive::{adaptive_pipeline_was_rejected, plan_adaptive_pipeline_diagnostics};
pub use strict_bounded::{
    NativeStrictBoundedAcceptedCandidateDiagnostics, NativeStrictBoundedCompiledGraphDiagnostics,
    NativeStrictBoundedContractDiagnostics, NativeStrictBoundedRejectedCandidateDiagnostics,
    STRICT_BOUNDED_NON_JOIN_OUTSIDE_REPORT_SCOPE_REASON,
};
use strict_bounded::{
    accepted_plan_contract_diagnostics, rejected_plan_contract_diagnostics, strict_bounded_admission_is_rejected,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativePlanningDiagnostics {
    pub input_plan_root: String,
    pub input_plan_shape: String,
    pub report: PlanningReport,
    pub native_optimizer: cudf_query_engine::NativeOptimizerDiagnostics,
    pub source_diagnostics: Vec<cudf_query_engine::SourceDiagnostics>,
    pub strict_bounded_contract: NativeStrictBoundedContractDiagnostics,
    pub adaptive_pipeline: NativeAdaptivePipelineDiagnostics,
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
    last_native_optimizer: Mutex<cudf_query_engine::NativeOptimizerDiagnostics>,
    last_source_diagnostics: Mutex<Vec<cudf_query_engine::SourceDiagnostics>>,
    last_strict_bounded_contract: Mutex<NativeStrictBoundedContractDiagnostics>,
    last_adaptive_pipeline: Mutex<NativeAdaptivePipelineDiagnostics>,
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
                last_native_optimizer: Mutex::new(cudf_query_engine::NativeOptimizerDiagnostics::default()),
                last_source_diagnostics: Mutex::new(Vec::new()),
                last_strict_bounded_contract: Mutex::new(NativeStrictBoundedContractDiagnostics::default()),
                last_adaptive_pipeline: Mutex::new(NativeAdaptivePipelineDiagnostics::default()),
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

        let native_execution_config = self.config.native_execution_config();
        let native_optimizer_options = self.config.native_optimizer_options();

        let optimized = match self.phase {
            NativeOptimizerPhase::CountOnlyProbe => {
                optimize_count_only_native_plan(Arc::clone(&plan), native_execution_config, native_optimizer_options)?
            },
            NativeOptimizerPhase::Final => {
                optimize_final_native_plan(Arc::clone(&plan), native_execution_config, native_optimizer_options)?
            },
        };
        let optimized = apply_native_fallback_policy(Arc::clone(&plan), optimized, &self.config)?;
        *self.state.last_report.lock().expect("native planning report mutex poisoned") = Some(optimized.report.clone());
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
        match self.phase {
            NativeOptimizerPhase::CountOnlyProbe => CUDF_NATIVE_COUNT_ONLY_PHYSICAL_OPTIMIZER_RULE_NAME,
            NativeOptimizerPhase::Final => CUDF_NATIVE_PHYSICAL_OPTIMIZER_RULE_NAME,
        }
    }

    fn schema_check(&self) -> bool {
        true
    }
}

fn apply_native_fallback_policy(
    original_plan: Arc<dyn ExecutionPlan>,
    attempt: NativeOptimizationAttempt,
    config: &CudfOptimizerConfig,
) -> DataFusionResult<NativeOptimizationAttempt> {
    match config.fallback_policy {
        GpuFallbackPolicy::GpuPreferred => Ok(attempt),
        GpuFallbackPolicy::CpuOnly => Ok(NativeOptimizationAttempt {
            plan: original_plan,
            report: native_cpu_only_report(&attempt.report),
            native_optimizer: attempt.native_optimizer,
            source_diagnostics: attempt.source_diagnostics,
            strict_bounded_contract: NativeStrictBoundedContractDiagnostics::default(),
            adaptive_pipeline: NativeAdaptivePipelineDiagnostics::default(),
            rewritten: false,
        }),
        GpuFallbackPolicy::GpuOnly => {
            if let Some((candidate, reason)) = first_native_rejection(&attempt.report) {
                return Err(candidate_rejected_error(candidate, reason, config).into());
            }
            Ok(attempt)
        },
    }
}

fn native_cpu_only_report(report: &PlanningReport) -> PlanningReport {
    PlanningReport::from_outcomes(report.outcomes().iter().map(|outcome| match outcome {
        CandidateOutcome::Rewritten { candidate } | CandidateOutcome::Rejected { candidate, .. } => {
            CandidateOutcome::Rejected { candidate: candidate.clone(), reason: FallbackReason::ExplicitCpuPolicy }
        },
    }))
}

fn first_native_rejection(report: &PlanningReport) -> Option<(&GpuCandidate, FallbackReason)> {
    report.outcomes().iter().find_map(|outcome| match outcome {
        CandidateOutcome::Rejected { candidate, reason } => Some((candidate, *reason)),
        CandidateOutcome::Rewritten { .. } => None,
    })
}

#[derive(Debug, Clone)]
struct NativeOptimizationAttempt {
    plan: Arc<dyn ExecutionPlan>,
    report: PlanningReport,
    native_optimizer: cudf_query_engine::NativeOptimizerDiagnostics,
    source_diagnostics: Vec<cudf_query_engine::SourceDiagnostics>,
    strict_bounded_contract: NativeStrictBoundedContractDiagnostics,
    adaptive_pipeline: NativeAdaptivePipelineDiagnostics,
    rewritten: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HostArrowChildReplacementMode {
    Normal,
    OuterJoinBoundary,
}

fn optimize_count_only_native_plan(
    plan: Arc<dyn ExecutionPlan>,
    native_execution_config: &NativeCudfExecutionConfig,
    native_optimizer_options: &cudf_query_engine::NativeOptimizerOptions,
) -> DataFusionResult<NativeOptimizationAttempt> {
    let lowering_input = peel_output_requirement_exec(Arc::clone(&plan));
    let lowering = crate::native::lower_count_only_physical_plan_to_native(lowering_input);
    replace_lowered_root_or_keep_plan(plan, lowering, native_execution_config, native_optimizer_options)
}

fn optimize_final_native_plan(
    plan: Arc<dyn ExecutionPlan>,
    native_execution_config: &NativeCudfExecutionConfig,
    native_optimizer_options: &cudf_query_engine::NativeOptimizerOptions,
) -> DataFusionResult<NativeOptimizationAttempt> {
    let lowering = crate::native::lower_physical_plan_to_native(Arc::clone(&plan));
    if lowering.plans.len() == 1 && lowering.rewritten_count() == 1 {
        let attempt = replace_lowered_root_or_keep_plan(
            Arc::clone(&plan),
            lowering,
            native_execution_config,
            native_optimizer_options,
        )?;
        if attempt.rewritten {
            return Ok(attempt);
        }

        let mut report = attempt.report.clone();
        let mut native_optimizer = attempt.native_optimizer.clone();
        let mut source_diagnostics = attempt.source_diagnostics.clone();
        let mut strict_bounded_contract = attempt.strict_bounded_contract.clone();
        let mut adaptive_pipeline = attempt.adaptive_pipeline.clone();
        let mode = if subtree_contains_outer_join(plan.as_ref()) {
            HostArrowChildReplacementMode::OuterJoinBoundary
        } else {
            HostArrowChildReplacementMode::Normal
        };
        let Some(rewritten_plan) = optimize_native_children_for_host_arrow_boundary(
            Arc::clone(&plan),
            native_execution_config,
            native_optimizer_options,
            mode,
            &mut report,
            &mut native_optimizer,
            &mut source_diagnostics,
            &mut strict_bounded_contract,
            &mut adaptive_pipeline,
        )?
        else {
            return Ok(attempt);
        };

        return Ok(NativeOptimizationAttempt {
            plan: rewritten_plan,
            report,
            native_optimizer,
            source_diagnostics,
            strict_bounded_contract,
            adaptive_pipeline,
            rewritten: true,
        });
    }

    let mut report = lowering.report.clone();
    let mut native_optimizer = cudf_query_engine::NativeOptimizerDiagnostics::default();
    let mut source_diagnostics = lowering.source_diagnostics.clone();
    let mut strict_bounded_contract = NativeStrictBoundedContractDiagnostics::default();
    let mut adaptive_pipeline = NativeAdaptivePipelineDiagnostics::default();
    let Some(rewritten_plan) = optimize_native_children_for_host_arrow_boundary(
        Arc::clone(&plan),
        native_execution_config,
        native_optimizer_options,
        HostArrowChildReplacementMode::Normal,
        &mut report,
        &mut native_optimizer,
        &mut source_diagnostics,
        &mut strict_bounded_contract,
        &mut adaptive_pipeline,
    )?
    else {
        return Ok(NativeOptimizationAttempt {
            plan,
            report,
            native_optimizer,
            source_diagnostics,
            strict_bounded_contract,
            adaptive_pipeline,
            rewritten: false,
        });
    };

    Ok(NativeOptimizationAttempt {
        plan: rewritten_plan,
        report,
        native_optimizer,
        source_diagnostics,
        strict_bounded_contract,
        adaptive_pipeline,
        rewritten: true,
    })
}

fn replace_lowered_root_or_keep_plan(
    plan: Arc<dyn ExecutionPlan>,
    lowering: crate::native::NativeLoweringResult,
    native_execution_config: &NativeCudfExecutionConfig,
    native_optimizer_options: &cudf_query_engine::NativeOptimizerOptions,
) -> DataFusionResult<NativeOptimizationAttempt> {
    if lowering.plans.len() != 1 || lowering.rewritten_count() != 1 {
        return Ok(NativeOptimizationAttempt {
            plan,
            report: lowering.report,
            native_optimizer: cudf_query_engine::NativeOptimizerDiagnostics::default(),
            source_diagnostics: lowering.source_diagnostics,
            strict_bounded_contract: NativeStrictBoundedContractDiagnostics::default(),
            adaptive_pipeline: NativeAdaptivePipelineDiagnostics::default(),
            rewritten: false,
        });
    }
    let candidate = single_rewritten_candidate(&lowering.report)
        .cloned()
        .ok_or_else(|| native_candidate_error("native rewrite has no rewritten candidate"))?;
    let optimizer_options = native_optimizer_options_for_execution(
        native_optimizer_options,
        native_execution_config.native_pipeline_execution_mode(),
    );
    let optimizer_result = cudf_query_engine::optimize_query_plan(lowering.plans[0].clone(), &optimizer_options);
    let native_optimizer = optimizer_result.diagnostics;
    let native_plan = optimizer_result.plan;

    if native_execution_config.native_pipeline_execution_mode()
        == cudf_query_engine::NativePipelineExecutionMode::Adaptive
    {
        let adaptive_pipeline =
            plan_adaptive_pipeline_diagnostics(&native_plan, candidate.clone(), native_execution_config)?
                .unwrap_or_default();
        if adaptive_pipeline_was_rejected(&adaptive_pipeline) {
            return Ok(NativeOptimizationAttempt {
                plan,
                report: native_execution_rejection_report(
                    &lowering.report,
                    FallbackReason::NativeAdaptivePipelineUnsupported,
                ),
                native_optimizer,
                source_diagnostics: lowering.source_diagnostics,
                strict_bounded_contract: NativeStrictBoundedContractDiagnostics::default(),
                adaptive_pipeline,
                rewritten: false,
            });
        }

        let plan = NativeCudfExec::try_new(native_plan, native_execution_config.clone())
            .map(|plan| Arc::new(plan) as Arc<dyn ExecutionPlan>)
            .map_err(DataFusionError::from)?;
        return Ok(NativeOptimizationAttempt {
            plan,
            report: lowering.report,
            native_optimizer,
            source_diagnostics: lowering.source_diagnostics,
            strict_bounded_contract: NativeStrictBoundedContractDiagnostics::default(),
            adaptive_pipeline,
            rewritten: true,
        });
    }

    if native_execution_boundary_is_rejected(&native_plan, native_execution_config)? {
        let report = native_execution_rejection_report(&lowering.report, FallbackReason::NativeBoundedJoinUnsupported);
        let candidate = first_bounded_join_unsupported_candidate(&report)
            .cloned()
            .ok_or_else(|| native_candidate_error("strict bounded rejection has no bounded-join candidate"))?;
        return Ok(NativeOptimizationAttempt {
            plan,
            report,
            native_optimizer,
            source_diagnostics: lowering.source_diagnostics,
            strict_bounded_contract: rejected_plan_contract_diagnostics(
                &native_plan,
                candidate,
                native_execution_config,
            )?,
            adaptive_pipeline: NativeAdaptivePipelineDiagnostics::default(),
            rewritten: false,
        });
    }

    let strict_bounded_contract = accepted_plan_contract_diagnostics(&native_plan, candidate, native_execution_config)?;
    let plan = NativeCudfExec::try_new(native_plan, native_execution_config.clone())
        .map(|plan| Arc::new(plan) as Arc<dyn ExecutionPlan>)
        .map_err(DataFusionError::from)?;
    Ok(NativeOptimizationAttempt {
        plan,
        report: lowering.report,
        native_optimizer,
        source_diagnostics: lowering.source_diagnostics,
        strict_bounded_contract,
        adaptive_pipeline: NativeAdaptivePipelineDiagnostics::default(),
        rewritten: true,
    })
}

fn native_optimizer_options_for_execution(
    options: &cudf_query_engine::NativeOptimizerOptions,
    execution_mode: cudf_query_engine::NativePipelineExecutionMode,
) -> cudf_query_engine::NativeOptimizerOptions {
    let mut options = options.clone();
    options.semijoin_reduction.admission_mode = match execution_mode {
        cudf_query_engine::NativePipelineExecutionMode::StrictBounded => {
            cudf_query_engine::SemijoinReductionAdmissionMode::StrictBounded
        },
        cudf_query_engine::NativePipelineExecutionMode::Adaptive => {
            cudf_query_engine::SemijoinReductionAdmissionMode::Adaptive
        },
    };
    options
}

fn optimize_native_children_for_host_arrow_boundary(
    plan: Arc<dyn ExecutionPlan>,
    native_execution_config: &NativeCudfExecutionConfig,
    native_optimizer_options: &cudf_query_engine::NativeOptimizerOptions,
    mode: HostArrowChildReplacementMode,
    report: &mut PlanningReport,
    native_optimizer: &mut cudf_query_engine::NativeOptimizerDiagnostics,
    source_diagnostics: &mut Vec<cudf_query_engine::SourceDiagnostics>,
    strict_bounded_contract: &mut NativeStrictBoundedContractDiagnostics,
    adaptive_pipeline: &mut NativeAdaptivePipelineDiagnostics,
) -> DataFusionResult<Option<Arc<dyn ExecutionPlan>>> {
    if !host_arrow_boundary_allows_child_replacement(plan.as_ref()) {
        return Ok(None);
    }
    if precision_sensitive_aggregate_requires_whole_subtree_rejection(plan.as_ref())
        || subtree_contains_unsafe_join_residual(plan.as_ref())
    {
        return Ok(None);
    }

    let children = plan.children();
    if children.is_empty() {
        return Ok(None);
    }

    let mut rewritten = false;
    let mut child_reports = Vec::new();
    let mut child_native_optimizer = cudf_query_engine::NativeOptimizerDiagnostics::default();
    let mut child_source_diagnostics = Vec::new();
    let mut new_children = Vec::with_capacity(children.len());
    for child in children {
        if window_boundary_should_preserve_sort_child(plan.as_ref(), child.as_ref()) {
            if let Some(rewritten_child) = optimize_native_children_for_host_arrow_boundary(
                Arc::clone(child),
                native_execution_config,
                native_optimizer_options,
                mode,
                report,
                native_optimizer,
                source_diagnostics,
                strict_bounded_contract,
                adaptive_pipeline,
            )? {
                rewritten = true;
                new_children.push(rewritten_child);
            } else {
                new_children.push(Arc::clone(child));
            }
            continue;
        }
        if mode == HostArrowChildReplacementMode::Normal
            && join_boundary_should_keep_datafusion_child(plan.as_ref(), child.as_ref())
        {
            new_children.push(Arc::clone(child));
            continue;
        }
        if subtree_contains_outer_join(child.as_ref())
            || (mode == HostArrowChildReplacementMode::OuterJoinBoundary && subtree_contains_join(child.as_ref()))
        {
            if let Some(rewritten_child) = optimize_native_children_for_host_arrow_boundary(
                Arc::clone(child),
                native_execution_config,
                native_optimizer_options,
                HostArrowChildReplacementMode::OuterJoinBoundary,
                report,
                native_optimizer,
                source_diagnostics,
                strict_bounded_contract,
                adaptive_pipeline,
            )? {
                rewritten = true;
                new_children.push(rewritten_child);
            } else {
                new_children.push(Arc::clone(child));
            }
            continue;
        }
        let attempt = optimize_final_native_plan(Arc::clone(child), native_execution_config, native_optimizer_options)?;
        if attempt.rewritten && !host_arrow_child_replacement_preserves_schema(child.as_ref(), attempt.plan.as_ref()) {
            new_children.push(Arc::clone(child));
            continue;
        }
        rewritten |= attempt.rewritten;
        child_reports.push(attempt.report);
        child_native_optimizer.merge(attempt.native_optimizer);
        child_source_diagnostics.extend(attempt.source_diagnostics);
        strict_bounded_contract.merge(attempt.strict_bounded_contract);
        adaptive_pipeline.merge(attempt.adaptive_pipeline);
        new_children.push(attempt.plan);
    }

    if !rewritten {
        return Ok(None);
    }

    for child_report in child_reports {
        append_planning_report(report, &child_report);
    }
    native_optimizer.merge(child_native_optimizer);
    source_diagnostics.extend(child_source_diagnostics);
    plan.with_new_children(new_children).map(Some)
}

fn host_arrow_child_replacement_preserves_schema(
    original: &dyn ExecutionPlan,
    replacement: &dyn ExecutionPlan,
) -> bool {
    original.schema().as_ref() == replacement.schema().as_ref()
}

fn append_planning_report(report: &mut PlanningReport, extra: &PlanningReport) {
    *report = PlanningReport::from_outcomes(report.outcomes().iter().cloned().chain(extra.outcomes().iter().cloned()));
}

fn host_arrow_boundary_allows_child_replacement(plan: &dyn ExecutionPlan) -> bool {
    plan.as_any().downcast_ref::<FilterExec>().is_some()
        || plan.as_any().downcast_ref::<ProjectionExec>().is_some()
        || plan.as_any().downcast_ref::<SortExec>().is_some()
        || plan.as_any().downcast_ref::<GlobalLimitExec>().is_some()
        || plan.as_any().downcast_ref::<LocalLimitExec>().is_some()
        || plan.as_any().downcast_ref::<AggregateExec>().is_some()
        || plan.as_any().downcast_ref::<CoalescePartitionsExec>().is_some()
        || plan.as_any().downcast_ref::<RepartitionExec>().is_some()
        || plan.as_any().downcast_ref::<HashJoinExec>().is_some()
        || plan.as_any().downcast_ref::<CrossJoinExec>().is_some()
        || plan.as_any().downcast_ref::<NestedLoopJoinExec>().is_some()
        || plan.as_any().downcast_ref::<UnionExec>().is_some()
        || plan.as_any().downcast_ref::<WindowAggExec>().is_some()
        || plan.as_any().downcast_ref::<BoundedWindowAggExec>().is_some()
}

fn join_boundary_should_keep_datafusion_child(parent: &dyn ExecutionPlan, child: &dyn ExecutionPlan) -> bool {
    if parent.as_any().downcast_ref::<HashJoinExec>().is_none()
        && parent.as_any().downcast_ref::<NestedLoopJoinExec>().is_none()
    {
        return false;
    }
    child.as_any().downcast_ref::<DataSourceExec>().is_some() || child.as_any().downcast_ref::<FilterExec>().is_some()
}

fn window_boundary_should_preserve_sort_child(parent: &dyn ExecutionPlan, child: &dyn ExecutionPlan) -> bool {
    (parent.as_any().downcast_ref::<WindowAggExec>().is_some()
        || parent.as_any().downcast_ref::<BoundedWindowAggExec>().is_some())
        && child.as_any().downcast_ref::<SortExec>().is_some()
}

fn precision_sensitive_aggregate_requires_whole_subtree_rejection(plan: &dyn ExecutionPlan) -> bool {
    let Some(sort) = plan.as_any().downcast_ref::<SortExec>() else {
        return false;
    };
    sort.fetch().is_none() && subtree_contains_precision_sensitive_unsupported_aggregate(plan)
}

fn subtree_contains_precision_sensitive_unsupported_aggregate(plan: &dyn ExecutionPlan) -> bool {
    if let Some(aggregate) = plan.as_any().downcast_ref::<AggregateExec>()
        && aggregate.aggr_expr().iter().any(|expr| {
            let function_name = expr.fun().name();
            function_name.eq_ignore_ascii_case("stddev") || function_name.eq_ignore_ascii_case("stddev_samp")
        })
    {
        return true;
    }
    plan.children().iter().any(|child| subtree_contains_precision_sensitive_unsupported_aggregate(child.as_ref()))
}

fn subtree_contains_unsafe_join_residual(plan: &dyn ExecutionPlan) -> bool {
    if let Some(join) = plan.as_any().downcast_ref::<HashJoinExec>()
        && join
            .filter()
            .is_some_and(|filter| join_residual_contains_nullable_scalar_or(filter.expression(), filter.schema()))
    {
        return true;
    }
    if let Some(join) = plan.as_any().downcast_ref::<NestedLoopJoinExec>()
        && join
            .filter()
            .is_some_and(|filter| join_residual_contains_nullable_scalar_or(filter.expression(), filter.schema()))
    {
        return true;
    }
    plan.children().iter().any(|child| subtree_contains_unsafe_join_residual(child.as_ref()))
}

fn subtree_contains_outer_join(plan: &dyn ExecutionPlan) -> bool {
    if let Some(join) = plan.as_any().downcast_ref::<HashJoinExec>()
        && matches!(join.join_type(), JoinType::Left | JoinType::Right | JoinType::Full)
    {
        return true;
    }
    if let Some(join) = plan.as_any().downcast_ref::<NestedLoopJoinExec>()
        && matches!(join.join_type(), JoinType::Left | JoinType::Right | JoinType::Full)
    {
        return true;
    }
    plan.children().iter().any(|child| subtree_contains_outer_join(child.as_ref()))
}

fn subtree_contains_join(plan: &dyn ExecutionPlan) -> bool {
    plan.as_any().downcast_ref::<HashJoinExec>().is_some()
        || plan.as_any().downcast_ref::<CrossJoinExec>().is_some()
        || plan.as_any().downcast_ref::<NestedLoopJoinExec>().is_some()
        || plan.children().iter().any(|child| subtree_contains_join(child.as_ref()))
}

fn join_residual_contains_nullable_scalar_or(expr: &Arc<dyn PhysicalExpr>, schema: &Schema) -> bool {
    if let Some(binary) = expr.as_any().downcast_ref::<BinaryExpr>()
        && matches!(binary.op(), datafusion::logical_expr::Operator::Or)
        && expr.nullable(schema).unwrap_or(true)
        && expr_contains_scalar_function(expr)
    {
        return true;
    }
    expr.children().into_iter().any(|child| join_residual_contains_nullable_scalar_or(child, schema))
}

fn expr_contains_scalar_function(expr: &Arc<dyn PhysicalExpr>) -> bool {
    expr.as_any().downcast_ref::<ScalarFunctionExpr>().is_some()
        || expr.children().into_iter().any(expr_contains_scalar_function)
}

fn native_execution_boundary_is_rejected(
    plan: &cudf_query_engine::QueryPlan,
    execution_config: &NativeCudfExecutionConfig,
) -> DataFusionResult<bool> {
    if !execution_config.memory_control_policy().is_bounded() || !plan.contains_join() {
        return Ok(false);
    }
    strict_bounded_admission_is_rejected(plan)
}

fn native_execution_rejection_report(report: &PlanningReport, rejection_reason: FallbackReason) -> PlanningReport {
    PlanningReport::from_outcomes(report.outcomes().iter().map(|outcome| match outcome {
        CandidateOutcome::Rewritten { candidate } => {
            CandidateOutcome::Rejected { candidate: candidate.clone(), reason: rejection_reason }
        },
        CandidateOutcome::Rejected { candidate, reason } => {
            CandidateOutcome::Rejected { candidate: candidate.clone(), reason: *reason }
        },
    }))
}

fn single_rewritten_candidate(report: &PlanningReport) -> Option<&GpuCandidate> {
    let mut rewritten = report.outcomes().iter().filter_map(|outcome| match outcome {
        CandidateOutcome::Rewritten { candidate } => Some(candidate),
        CandidateOutcome::Rejected { .. } => None,
    });
    let candidate = rewritten.next()?;
    rewritten.next().is_none().then_some(candidate)
}

fn first_bounded_join_unsupported_candidate(report: &PlanningReport) -> Option<&GpuCandidate> {
    report.outcomes().iter().find_map(|outcome| match outcome {
        CandidateOutcome::Rejected { candidate, reason } if *reason == FallbackReason::NativeBoundedJoinUnsupported => {
            Some(candidate)
        },
        CandidateOutcome::Rejected { .. } | CandidateOutcome::Rewritten { .. } => None,
    })
}

fn native_candidate_error(message: &'static str) -> DataFusionError {
    Error::new(
        crate::error::ErrorCode::InvalidExecutionPlan,
        "datafusion_cudf.native.pipeline_candidate_diagnostics",
        message,
    )
    .into()
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
